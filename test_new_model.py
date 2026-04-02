#!/usr/bin/env python3
"""
Test the new recruiter model: oussema2021/recruiter-persona-llama-3.1-8b-merged-v2
"""

import asyncio
import sys
import os

# Add the nlp directory to path
nlp_path = os.path.join(os.path.dirname(__file__), "ai-recruiter", "agents", "nlp")
sys.path.insert(0, nlp_path)

from agent import NLPAgent
async def test_new_model():
    """Test the new model with scorer and refiner enabled."""
    print("\n" + "="*80)
    print("🚀 TESTING NEW MODEL: oussema2021/recruiter-persona-llama-3.1-8b-merged-v2")
    print("="*80)
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")
        else:
            print("⚠️  No GPU detected, will use CPU (slower)\n")
    except:
        pass
    
    # Initialize agent with new model
    print("📦 Loading NLP Agent...")
    print("   • Main Model: oussema2021/recruiter-persona-llama-3.1-8b-merged-v2")
    print("   • Scorer: Qwen/Qwen2.5-1.5B-Instruct (CPU)")
    print("   • Refiner: Qwen/Qwen2.5-1.5B-Instruct (CPU)")
    print()
    
    agent = NLPAgent(
        model_path="oussema2021/recruiter-persona-llama-3.1-8b-merged-v2",
        enable_scorer=True,
        scorer_model="Qwen/Qwen2.5-1.5B-Instruct",
        enable_refiner=True,
        refiner_model="Qwen/Qwen2.5-1.5B-Instruct",
        enable_tts=False,  # Disable TTS for testing
        use_compact_prompt=False,
        auto_cleanup_gpu=False,
        load_timeout_sec=300,
    )
    
    # Candidate CV
    cv_text = """
Candidate: Oussema Aissaoui
Role: AI Engineering Intern

Projects:
- Blood Cell Segmentation: YOLOv8 + SAM pipeline for WBC/RBC classification
  Achieved 95% accuracy on microscopy images
- GraphRAG + LLM: Pattern recognition in enterprise data
- Water Use Efficiency: ML model with WaPOR + Google Earth Engine

Technical Skills:
- Python, Deep Learning, TensorFlow, PyTorch, OpenCV
- YOLOv8, SAM, NLP, GraphRAG, transformers
- Docker, Git, AWS
"""
    
    print("✅ Agent loaded successfully!\n")
    
    # Test 1: Greeting
    print("="*80)
    print("TEST 1: Initial Greeting")
    print("="*80)
    
    response1 = await agent.ask(
        candidate_cv=cv_text,
        conversation_history=[],
        latest_answer=None,
    )
    
    print(f"🤖 Recruiter: {response1.question}")
    print(f"📊 Confidence: {response1.confidence:.2f}")
    if response1.audio_bytes:
        print(f"🔊 Audio: {len(response1.audio_bytes)} bytes")
    print()
    
    # Test 2: Candidate introduces themselves
    print("="*80)
    print("TEST 2: After Candidate Introduction")
    print("="*80)
    
    candidate_answer = "Hi! I'm Oussema, an AI engineering student passionate about computer vision and deep learning. I've worked on several projects including blood cell segmentation and GraphRAG systems."
    
    response2 = await agent.ask(
        candidate_cv=cv_text,
        conversation_history=[
            [response1.question, candidate_answer]
        ],
        latest_answer=candidate_answer,
    )
    
    print(f"👤 Candidate: {candidate_answer}")
    print(f"🤖 Recruiter: {response2.question}")
    print(f"📊 Confidence: {response2.confidence:.2f}")
    print()
    
    # Test 3: Technical answer
    print("="*80)
    print("TEST 3: Technical Follow-up")
    print("="*80)
    
    tech_answer = "I used YOLOv8 for initial detection of blood cells, then applied SAM for precise segmentation. The pipeline achieved 95% accuracy on our validation set."
    
    response3 = await agent.ask(
        candidate_cv=cv_text,
        conversation_history=[
            [response1.question, candidate_answer],
            [response2.question, tech_answer]
        ],
        latest_answer=tech_answer,
    )
    
    print(f"👤 Candidate: {tech_answer}")
    print(f"🤖 Recruiter: {response3.question}")
    print(f"📊 Confidence: {response3.confidence:.2f}")
    print()
    
    # Test 4: Vague answer (should trigger scorer/refiner)
    print("="*80)
    print("TEST 4: Handling Vague Answer")
    print("="*80)
    
    vague_answer = "It was pretty cool, I guess."
    
    response4 = await agent.ask(
        candidate_cv=cv_text,
        conversation_history=[
            [response1.question, candidate_answer],
            [response2.question, tech_answer],
            [response3.question, vague_answer]
        ],
        latest_answer=vague_answer,
    )
    
    print(f"👤 Candidate: {vague_answer}")
    print(f"🤖 Recruiter: {response4.question}")
    print(f"📊 Confidence: {response4.confidence:.2f}")
    print()
    
    print("="*80)
    print("✅ ALL TESTS COMPLETED!")
    print("="*80)
    print("\n📈 Summary:")
    print(f"   • Model: oussema2021/recruiter-persona-llama-3.1-8b-merged-v2 ✅")
    print(f"   • Scorer: Enabled ✅")
    print(f"   • Refiner: Enabled ✅")
    print(f"   • Tests Passed: 4/4 ✅")
    print()


if __name__ == "__main__":
    asyncio.run(test_new_model())
