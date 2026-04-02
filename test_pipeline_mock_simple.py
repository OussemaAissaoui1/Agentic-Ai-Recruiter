#!/usr/bin/env python3
"""
Simple test of scorer and refiner with increased timeout.
Tests the actual question you provided.
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


async def test_with_patience():
    """Test with longer wait times to let CPU models complete."""
    
    print("\n" + "="*80)
    print("🧪 PIPELINE TEST - Your Mock Question")
    print("="*80)
    
    # Your exact question
    question = "That's great! Now, let's dive into another project that caught our attention – the Blood Cell Segmentation project using YOLOv8 and SAM pipeline. Can you walk us through how you approached this challenge, specifically focusing on the?"
    
    # Mock candidate answer (the one being scored)
    answer = "I combined YOLOv8 for object detection with SAM for precise segmentation. The pipeline processes microscopy images to classify white and red blood cells."
    
    cv_text = """
Candidate: Oussema Aissaoui
Projects:
- Blood Cell Segmentation: YOLOv8 + SAM pipeline for WBC/RBC classification
- GraphRAG + LLM reasoning for pattern recognition
"""
    
    history = [
        None ]
    
    print(f"\n❓ Question to Refine:")
    print(f'   "{question}"')
    print(f"\n💬 Answer Being Scored:")
    print(f'   "{answer}"')
    
    # =========================================================================
    # SCORER
    # =========================================================================
    print("\n" + "="*80)
    print("🧠 SCORER - Analyzing answer (this may take 5-10 seconds on CPU)...")
    print("="*80)
    
    scorer = ResponseScorer()
    start = time.time()
    
    # Run scorer (with internal 3s timeout, but we'll wait longer externally)
    try:
        scorer_result = await asyncio.wait_for(
            scorer.score(
                candidate_cv=cv_text,
                conversation_history=history,
                latest_answer=answer,
                device="cuda"  # Force CPU to test patience
            ),  
            timeout=10.0  # Give it 10 seconds total
        )
        
        elapsed = time.time() - start
        print(f"\n✅ Scorer completed in {elapsed:.1f}s")
        print(f"\n📊 Analysis:")
        print(f"   • Answer Quality: {scorer_result.answer_quality}")
        print(f"   • Engagement: {scorer_result.candidate_engagement}")
        print(f"   • Knowledge Level: {scorer_result.knowledge_level}")
        print(f"   • Suggested Action: {scorer_result.suggested_action}")
        print(f"   • Question Focus: {scorer_result.question_focus}")
        print(f"   • Current Topic: {scorer_result.current_topic}")
        
        if scorer_result.acknowledgment:
            print(f"\n💬 Suggested Acknowledgment: \"{scorer_result.acknowledgment}\"")
        
        if scorer_result.knowledge_gaps:
            print(f"\n🔍 Knowledge Gaps:")
            for gap in scorer_result.knowledge_gaps:
                if gap and gap != "timeout":
                    print(f"   - {gap}")
        
    except asyncio.TimeoutError:
        print("\n⏱️  Scorer timed out after 10s (CPU too slow)")
        scorer_result = None
    except Exception as e:
        print(f"\n❌ Scorer error: {e}")
        import traceback
        traceback.print_exc()
        scorer_result = None
    
    # =========================================================================
    # REFINER  
    # =========================================================================
    print("\n" + "="*80)
    print("✨ REFINER - Enhancing question (this may take 5-10 seconds on CPU)...")
    print("="*80)
    
    refiner = QuestionRefiner()
    start = time.time()
    
    try:
        refiner_result = await asyncio.wait_for(
            refiner.refine(
                original_question=question,
                cv_topic=scorer_result.question_focus if scorer_result else "blood cell segmentation",
                knowledge_level=scorer_result.knowledge_level if scorer_result else "intermediate",
                response_quality=scorer_result.answer_quality if scorer_result else "detailed",
            ),
            timeout=10.0  # Give it 10 seconds
        )
        
        elapsed = time.time() - start
        print(f"\n✅ Refiner completed in {elapsed:.1f}s")
        print(f"\n📊 Results:")
        print(f"   • Refinement Type: {refiner_result.refinement_type}")
        
        if refiner_result.improvements and refiner_result.improvements != "timeout":
            print(f"   • Improvements: {refiner_result.improvements}")
        
        print(f"\n📝 ORIGINAL:")
        print(f'   "{question}"')
        print(f"\n✏️  REFINED:")
        print(f'   "{refiner_result.refined_question}"')
        
        # Show if anything changed
        if refiner_result.refined_question != question:
            print(f"\n✅ Question was improved!")
        else:
            print(f"\n ℹ️ Question maintained (already good or timeout)")
        
    except asyncio.TimeoutError:
        print("\n⏱️  Refiner timed out after 10s (CPU too slow)")
        refiner_result = None
    except Exception as e:
        print(f"\n❌ Refiner error: {e}")
        import traceback
        traceback.print_exc()
        refiner_result = None
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
    
    if scorer_result and refiner_result:
        print("\n🎯 Full Pipeline Success!")
        print(f"\n   Scorer says: {scorer_result.suggested_action}")
        print(f"   Refiner says: {refiner_result.refinement_type}")
    else:
        print("\n⚠️  Some components timed out (CPU may be too slow)")
        print("   This is normal for CPU-only inference on large models.")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(test_with_patience())
