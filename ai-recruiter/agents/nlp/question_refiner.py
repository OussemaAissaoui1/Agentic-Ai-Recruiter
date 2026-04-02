"""
Question Refiner — Small LLM for Question Enhancement

Takes a generated interview question + response analysis and refines it to be:
- Deeper and more probing
- Unique and specific (not generic)
- Maintaining warm, professional recruiter tone
- On-topic and relevant to the candidate's CV

Architecture:
    Generated Question + Scorer Analysis → Small LLM (CPU) → Refined Question

Performance:
- Qwen2.5-1.5B: ~400-800ms on CPU
- Runs in asyncio thread pool to avoid blocking
- Can be disabled for faster responses (uses original question)
"""

from __future__ import annotations

import os
import json
import re
import asyncio
import threading
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Set HuggingFace Hub timeout
os.environ.setdefault("HUGGINGFACE_HUB_HTTP_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")


@dataclass(frozen=True)
class RefinerOutput:
    """Refined question with metadata."""
    refined_question: str
    refinement_type: str  # "deepened", "specified", "rephrased", "maintained"
    improvements: str  # What was improved
    latency_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "refined_question": self.refined_question,
            "refinement_type": self.refinement_type,
            "improvements": self.improvements,
            "latency_ms": self.latency_ms,
        }


def _run_with_timeout(func, timeout_sec: int, error_msg: str = "Operation timed out"):
    """Run func in a daemon thread with timeout."""
    result_holder: list = [None]
    exc_holder: list = [None]

    def _wrapper():
        try:
            result_holder[0] = func()
        except Exception as exc:
            exc_holder[0] = exc

    t = threading.Thread(target=_wrapper, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        raise TimeoutError(error_msg)
    if exc_holder[0] is not None:
        raise exc_holder[0]
    return result_holder[0]


class QuestionRefiner:
    """
    Small LLM that refines interview questions on CPU.
    
    Takes a generated question and scorer analysis, then:
    - Makes it deeper and more probing
    - Ensures specificity (references CV items)
    - Maintains warm, professional tone
    - Removes generic phrasing
    """

    REFINER_PROMPT = """You are polishing a recruiter's interview question to sound more natural and human.

## Context:
CV Topic: {cv_topic}
Knowledge Level: {knowledge_level}
Response Quality: {response_quality}
Focus: {question_focus}
Acknowledgment hint: {acknowledgment}

## Original Question:
{original_question}

## Your Task:
Make it sound like a real person talking — warm, curious, specific. Keep the same topic and intent.

## STRICT Rules:
1. GREETING/INTRO → Keep as-is. Maybe add slight warmth.
2. SPECIFIC TOPIC → Keep that EXACT topic. Add technical specificity from the CV context.
3. INCOMPLETE/BROKEN → Fix it naturally while keeping the intent.
4. If acknowledgment is provided, weave it in naturally at the start (don't just paste it).
5. NEVER replace the topic. NEVER change the subject.
6. NEVER invent facts the candidate didn't mention. Don't say "you mentioned X" unless it was in the original question.
7. NEVER write as the candidate. You are outputting the RECRUITER's question.
8. NEVER use placeholder brackets like [specific language] or [topic]. Use concrete words only.
9. Output must end with exactly one "?"
10. Output must be under 200 characters.
11. Sound like a curious colleague, not a script-reader.

## Good Examples:
- "Oh nice, so you used Neo4j for the graph layer. How did you handle the query performance when the dataset scaled up?"
- "SAM for boundary detection — that's not trivial. What was the trickiest part of getting the segmentation pipeline to work end-to-end?"
- "Huh, 30+ screens is a lot. What was your approach for keeping the integrations consistent across all of them?"

## Bad Examples (NEVER do these):
- "Got it. Tell me about your project." (too robotic, too generic)
- "Can you explain the SAM pipeline?" (replaced a greeting with a technical question)
- "That's great! What challenges did you face?" (generic, could apply to anything)
- "You mentioned using TensorFlow..." (inventing what candidate said)
- "During my time at TALAN..." (writing as the CANDIDATE, not the recruiter!)

## If already good: set refinement_type to "maintained".

Respond with ONLY valid JSON:
{{
    "refined_question": "the polished question (MUST end with ? and be under 200 chars)",
    "refinement_type": "<pick ONE: deepened OR specified OR rephrased OR maintained>",
    "improvements": "what changed OR 'none'"
}}

JSON response:"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cpu",
        load_timeout_sec: int = 120,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._load_timeout = load_timeout_sec
        self._model = None
        self._tokenizer = None
        self._is_ready = False
        self._load_lock = threading.Lock()
        self._executor = None

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def _lazy_load(self) -> None:
        """Load model and tokenizer on first use."""
        if self._is_ready:
            return
            
        with self._load_lock:
            if self._is_ready:
                return
                
            print(f"[refiner] Loading {self._model_name} on {self._device}...")
            
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                def _load():
                    tokenizer = AutoTokenizer.from_pretrained(
                        self._model_name,
                        trust_remote_code=True,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        self._model_name,
                        torch_dtype=torch.float32 if self._device == "cpu" else torch.float16,
                        device_map=self._device,
                        trust_remote_code=True,
                    )
                    model.eval()
                    return tokenizer, model
                
                self._tokenizer, self._model = _run_with_timeout(
                    _load,
                    self._load_timeout,
                    f"Refiner model loading timed out after {self._load_timeout}s"
                )
                
                # Create thread pool for async inference
                from concurrent.futures import ThreadPoolExecutor
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="refiner")
                
                self._is_ready = True
                print(f"[refiner] Ready ({self._model_name})")
                
            except Exception as e:
                print(f"[refiner] Load failed: {e}")
                self._is_ready = False

    async def refine(
        self,
        original_question: str,
        cv_topic: str = "general CV items",
        knowledge_level: str = "uncertain",
        response_quality: str = "unknown",
        question_focus: str = "",
        acknowledgment: str = "",
        timeout_sec: float = 5.0,  # Increased from 3.0s
    ) -> RefinerOutput:
        """
        Refine a question using the small LLM.
        
        Returns refined question or original if refinement fails/times out.
        """
        # CRITICAL: Don't refine greetings/introductions - preserve them exactly
        lower_q = original_question.lower()
        is_greeting = any(phrase in lower_q for phrase in [
            'hello', 'hi ', 'welcome', 'good morning', 'good afternoon',
            'introduce yourself', 'tell me about yourself', 'my name is',
            'thanks for joining', 'pleasure to meet'
        ])
        
        if is_greeting:
            # Preserve greetings exactly - they're already well-crafted
            # Note: acknowledgments don't make sense before greetings, so skip them
            return RefinerOutput(
                refined_question=original_question,
                refinement_type="maintained",
                improvements="greeting_preserved",
                latency_ms=0.0,
            )
        
        if not self.is_ready:
            self._lazy_load()
            
        if not self.is_ready:
            # Model didn't load, return original
            return RefinerOutput(
                refined_question=original_question,
                refinement_type="maintained",
                improvements="refiner_unavailable",
                latency_ms=0.0,
            )
        
        try:
            t0 = time.perf_counter()
            
            # Build prompt
            prompt = self.REFINER_PROMPT.format(
                cv_topic=cv_topic,
                knowledge_level=knowledge_level,
                response_quality=response_quality,
                question_focus=question_focus or "technical depth",
                acknowledgment=acknowledgment or "(none - start directly)",
                original_question=original_question,
            )
            
            # Run inference in thread pool with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._generate, prompt),
                timeout=timeout_sec
            )
            
            latency = (time.perf_counter() - t0) * 1000
            
            # Parse JSON response
            try:
                # Try to extract JSON if wrapped in other text
                json_str = result
                if "{" in result and "}" in result:
                    start = result.find("{")
                    end = result.rfind("}") + 1
                    json_str = result[start:end]
                
                data = json.loads(json_str)
                refined = data.get("refined_question", "").strip()
                
                # Validation: must be non-empty, end with ?, be reasonable length, and not contain brackets
                has_brackets = bool(re.search(r"\[(?!http)[^\]]+\]", refined))
                if refined and refined.endswith("?") and 10 < len(refined) < 200 and not has_brackets:
                    # IMPORTANT: Prepend acknowledgment if provided and not already present
                    if acknowledgment and acknowledgment.strip():
                        ack = acknowledgment.strip()
                        # Check if acknowledgment is already in the refined question
                        if not refined.lower().startswith(ack.lower()):
                            # Add acknowledgment with proper spacing
                            refined = f"{ack} {refined}"
                    
                    return RefinerOutput(
                        refined_question=refined,
                        refinement_type=data.get("refinement_type", "refined"),
                        improvements=data.get("improvements", ""),
                        latency_ms=latency,
                    )
                else:
                    print(f"[refiner] Validation failed - refined='{refined}' (len={len(refined)}, ends_with_?={refined.endswith('?') if refined else False})")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[refiner] Parse error: {e}")
                print(f"[refiner] Raw output: {result[:500]}")
            
            # Fallback to original
            return RefinerOutput(
                refined_question=original_question,
                refinement_type="maintained",
                improvements="refinement_failed",
                latency_ms=latency,
            )
            
        except asyncio.TimeoutError:
            print(f"[refiner] Timeout after {timeout_sec}s")
            return RefinerOutput(
                refined_question=original_question,
                refinement_type="maintained",
                improvements="timeout",
                latency_ms=timeout_sec * 1000,
            )
        except Exception as e:
            print(f"[refiner] Error: {e}")
            return RefinerOutput(
                refined_question=original_question,
                refinement_type="maintained",
                improvements=f"error: {str(e)[:50]}",
                latency_ms=0.0,
            )

    def _generate(self, prompt: str) -> str:
        """Synchronous generation (runs in thread pool)."""
        import torch
        
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=200,  # Increased from 150 for longer responses
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        
        generated = self._tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated.strip()

    def shutdown(self) -> None:
        """Cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._model = None
        self._tokenizer = None
        self._is_ready = False
