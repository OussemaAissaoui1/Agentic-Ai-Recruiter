#!/usr/bin/env python3
"""
GPU-Accelerated Mock Test - Fast scorer and refiner test using GPU
"""

import asyncio
import sys
import os
import time

# Add the nlp directory to path
nlp_path = os.path.join(os.path.dirname(__file__), "ai-recruiter", "agents", "nlp")
sys.path.insert(0, nlp_path)

from response_scorer import ResponseScorer
from question_refiner import QuestionRefiner


async def test_with_gpu():
    """Test with GPU acceleration - should complete in <1 second each!"""
    
    print("\n" + "="*80)
    print("🚀 GPU-ACCELERATED PIPELINE TEST")
    print("="*80)
    
    # Your exact question
    question = "Hello! Welcome to our live interview today. My name is Alex, and I'm excited to speak with you about the AI Engineering Intern position. Can you please introduce yourself and tell me why you're interested in this role?"
    
    # Mock candidate answer
    answer = "bro wtf are you saying and who are you ???"
    
    cv_text = """
Candidate: Oussema Aissaoui
Role: AI Engineering Intern

Projects:
- Blood Cell Segmentation: YOLOv8 + SAM pipeline for WBC/RBC classification
  Achieved 95% accuracy on microscopy images
- GraphRAG + LLM: Pattern recognition in enterprise data
- Water Use Efficiency: ML model with WaPOR + Google Earth Engine

Technical Skills:
- Python, Deep Learning, TensorFlow, OpenCV, YOLOv8, SAM, NLP, GraphRAG
"""
    
    history = [
        ["Tell me about your projects.", "I've worked on several AI projects including GraphRAG."]
    ]
    
    print(f"\n❓ Question to Refine:")
    print(f'   "{question}"')
    print(f"\n💬 Answer Being Scored:")
    print(f'   "{answer}"')
    
    # =========================================================================
    # SCORER ON GPU
    # =========================================================================
    print("\n" + "="*80)
    print("🧠 SCORER - Analyzing answer (GPU-accelerated)")
    print("="*80)
    
    # Initialize scorer with GPU
    scorer = ResponseScorer(device="cuda")
    start = time.time()
    
    try:
        scorer_result = await asyncio.wait_for(
            scorer.score(
                candidate_cv=cv_text,
                conversation_history=history,
                latest_answer=answer,
            ),
            timeout=10.0
        )
        
        elapsed = time.time() - start
        print(f"\n✅ Scorer completed in {elapsed:.1f}s")
        print(f"\n📊 Full Analysis:")
        print(f"   • Answer Quality: {scorer_result.answer_quality}")
        print(f"   • Engagement: {scorer_result.candidate_engagement}")
        print(f"   • Knowledge Level: {scorer_result.knowledge_level}")
        print(f"   • Suggested Action: {scorer_result.suggested_action}")
        print(f"   • Question Focus: {scorer_result.question_focus}")
        print(f"   • Current Topic: {scorer_result.current_topic}")
        
        if scorer_result.acknowledgment:
            print(f"\n💬 Suggested Acknowledgment:")
            print(f'   "{scorer_result.acknowledgment}"')
        
        if scorer_result.knowledge_gaps:
            print(f"\n🔍 Knowledge Gaps Identified:")
            for gap in scorer_result.knowledge_gaps:
                if gap and gap != "timeout":
                    print(f"   - {gap}")
        
        if scorer_result.raw_reasoning:
            print(f"\n🧠 Scorer's Reasoning:")
            print(f"   {scorer_result.raw_reasoning[:200]}...")
        
    except asyncio.TimeoutError:
        print("\n⏱️  Scorer timed out after 10s")
        scorer_result = None
    except Exception as e:
        print(f"\n❌ Scorer error: {e}")
        import traceback
        traceback.print_exc()
        scorer_result = None
    
    # =========================================================================
    # REFINER ON GPU
    # =========================================================================
    print("\n" + "="*80)
    print("✨ REFINER - Enhancing question (GPU-accelerated)")
    print("="*80)
    
    # Initialize refiner with GPU
    refiner = QuestionRefiner(device="cuda")
    start = time.time()
    
    try:
        refiner_result = await asyncio.wait_for(
            refiner.refine(
                original_question=question,
                cv_topic=scorer_result.question_focus if scorer_result else "blood cell segmentation project",
                knowledge_level=scorer_result.knowledge_level if scorer_result else "intermediate",
                response_quality=scorer_result.answer_quality if scorer_result else "detailed",
                question_focus=scorer_result.question_focus if scorer_result else "technical details",
                acknowledgment=scorer_result.acknowledgment if scorer_result else "",
            ),
            timeout=10.0
        )
        
        elapsed = time.time() - start
        print(f"\n✅ Refiner completed in {elapsed:.1f}s")
        print(f"\n📊 Refinement Results:")
        print(f"   • Refinement Type: {refiner_result.refinement_type}")
        print(f"   • Latency: {refiner_result.latency_ms:.0f}ms")
        
        if refiner_result.improvements and refiner_result.improvements != "timeout":
            print(f"   • Improvements: {refiner_result.improvements}")
        
        print(f"\n📝 ORIGINAL QUESTION:")
        print(f'   "{question}"')
        print(f'   ({len(question)} characters)')
        
        print(f"\n✏️  REFINED QUESTION:")
        print(f'   "{refiner_result.refined_question}"')
        print(f'   ({len(refiner_result.refined_question)} characters)')
        
        # Show differences
        if refiner_result.refined_question != question:
            print(f"\n✅ Question was improved!")
            print(f"   • Type: {refiner_result.refinement_type}")
            print(f"   • Change: {len(refiner_result.refined_question) - len(question):+d} characters")
        else:
            print(f"\n ℹ️ Question maintained (already good)")
        
    except asyncio.TimeoutError:
        print("\n⏱️  Refiner timed out after 10s")
        refiner_result = None
    except Exception as e:
        print(f"\n❌ Refiner error: {e}")
        import traceback
        traceback.print_exc()
        refiner_result = None
    
    # =========================================================================
    # COMPLETE PIPELINE OUTPUT
    # =========================================================================
    print("\n" + "="*80)
    print("🎯 COMPLETE PIPELINE OUTPUT")
    print("="*80)
    
    if scorer_result and refiner_result:
        # Build the complete recruiter response
        ack = scorer_result.acknowledgment or None
        question_text = refiner_result.refined_question
        
        complete_response = f"{ack} {question_text}"
        
        print(f"\n💬 Full Recruiter Response:")
        print(f'   "{complete_response}"')
        
        print(f"\n📈 Pipeline Summary:")
        print(f"   • Answer was: {scorer_result.answer_quality} quality, {scorer_result.candidate_engagement} engagement")
        print(f"   • Suggested: {scorer_result.suggested_action}")
        print(f"   • Focus on: {scorer_result.question_focus}")
        print(f"   • Question: {refiner_result.refinement_type}")
        print(f"   • Improvements: {refiner_result.improvements}")
        
        print(f"\n⚡ Performance:")
        print(f"   • Scorer: {scorer_result.latency_ms:.0f}ms")
        print(f"   • Refiner: {refiner_result.latency_ms:.0f}ms")
        print(f"   • Total: {scorer_result.latency_ms + refiner_result.latency_ms:.0f}ms")
        
    else:
        print("\n⚠️  Some components failed - check errors above")
    
    print("\n" + "="*80)
    print("✅ GPU TEST COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🎮 GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")
        else:
            print("⚠️  No GPU detected, falling back to CPU (will be slower)\n")
    except:
        pass
    
    asyncio.run(test_with_gpu())
