#!/usr/bin/env python3
"""
Test the scorer and refiner with mock data.
Shows how the pipeline processes candidate answers and questions.
"""

import asyncio
import sys
import os

# Add the ai-recruiter/agents/nlp directory to path
nlp_path = os.path.join(os.path.dirname(__file__), "ai-recruiter", "agents", "nlp")
sys.path.insert(0, nlp_path)

from response_scorer import ResponseScorer
from question_refiner import QuestionRefiner


async def test_pipeline():
    print("\n" + "="*70)
    print("🧪 PIPELINE TEST - Mock Scenario")
    print("="*70)
    
    # Initialize components
    print("\n📦 Loading models...")
    scorer = ResponseScorer()
    refiner = QuestionRefiner()
    
    # Mock scenario
    cv_context = """
Candidate: Oussema Aissaoui
Role: AI Engineering Intern

Projects:
- Blood Cell Segmentation: YOLOv8 + SAM pipeline for WBC/RBC classification
- Pattern Recognition: GraphRAG + LLM reasoning
- Water Use Efficiency: ML model using WaPOR + Google Earth Engine

Technical Skills:
- Python, Deep Learning, TensorFlow, OpenCV, YOLOv8, SAM, NLP
"""
    
    # Candidate's answer to a previous question about their projects
    candidate_answer = """
Yes, I worked on several machine learning projects during my internships. 
The most interesting was the blood cell segmentation where I combined 
YOLOv8 for detection with SAM for precise segmentation. I used Python 
and trained the model on a dataset of microscopy images.
"""
    
    # The question we want to refine
    original_question = """That's great! Now, let's dive into another project that caught our attention – the Blood Cell Segmentation project using YOLOv8 and SAM pipeline. Can you walk us through how you approached this challenge, specifically focusing on the?"""
    
    conversation_history = [
        ["Tell me about your recent projects.", "Yes, I worked on several machine learning projects..."]
    ]
    
    print("\n" + "="*70)
    print("📝 TEST SCENARIO")
    print("="*70)
    print(f"\n💬 Candidate's Previous Answer:")
    print(f'   "{candidate_answer.strip()}"')
    print(f"\n❓ Original Question to Refine:")
    print(f'   "{original_question.strip()}"')
    
    # =========================================================================
    # STEP 1: Run Scorer on the candidate's answer
    # =========================================================================
    print("\n" + "="*70)
    print("🧠 STEP 1: SCORER - Analyzing candidate answer")
    print("="*70)
    
    try:
        scorer_output = await scorer.score(
            candidate_cv=cv_context,
            conversation_history=conversation_history,
            latest_answer=candidate_answer,
        )
        
        print(f"\n✓ Scorer Analysis Complete!")
        print(f"\n📊 Results:")
        print(f"   • Answer Quality: {scorer_output.answer_quality}")
        print(f"   • Engagement: {scorer_output.candidate_engagement}")
        print(f"   • Knowledge Level: {scorer_output.knowledge_level}")
        print(f"   • Suggested Action: {scorer_output.suggested_action}")
        print(f"   • Question Focus: {scorer_output.question_focus}")
        print(f"   • Current Topic: {scorer_output.current_topic}")
        
        if scorer_output.knowledge_gaps:
            print(f"\n🔍 Knowledge Gaps Identified:")
            for gap in scorer_output.knowledge_gaps:
                print(f"   - {gap}")
        
        if scorer_output.acknowledgment:
            print(f"\n💬 Suggested Acknowledgment:")
            print(f'   "{scorer_output.acknowledgment}"')
        
    except Exception as e:
        print(f"\n✗ Scorer Error: {e}")
        import traceback
        traceback.print_exc()
        scorer_output = None
    
    # =========================================================================
    # STEP 2: Run Refiner on the question
    # =========================================================================
    print("\n" + "="*70)
    print("✨ STEP 2: REFINER - Enhancing question")
    print("="*70)
    
    try:
        refiner_output = await refiner.refine(
            original_question=original_question,
            cv_topic=scorer_output.question_focus if scorer_output else "projects",
            knowledge_level=scorer_output.knowledge_level if scorer_output else "uncertain",
            response_quality=scorer_output.answer_quality if scorer_output else "unknown",
        )
        
        print(f"\n✓ Refinement Complete!")
        print(f"\n📊 Results:")
        print(f"   • Refinement Type: {refiner_output.refinement_type}")
        print(f"   • Latency: {refiner_output.latency_ms:.0f}ms")
        
        if refiner_output.improvements:
            print(f"\n💡 Improvements Made:")
            for improvement in refiner_output.improvements:
                print(f"   - {improvement}")
        
        print(f"\n📝 Original Question:")
        print(f'   "{original_question.strip()}"')
        
        print(f"\n✏️  Refined Question:")
        print(f'   "{refiner_output.refined_question.strip()}"')
        
        # Show comparison
        print(f"\n🔄 Comparison:")
        print(f"   Original length: {len(original_question)} chars")
        print(f"   Refined length:  {len(refiner_output.refined_question)} chars")
        
        if refiner_output.refinement_type != "maintained":
            print(f"   ✅ Question was enhanced!")
        else:
            print(f"   ℹ️  Original question was already good")
        
    except Exception as e:
        print(f"\n✗ Refiner Error: {e}")
        import traceback
        traceback.print_exc()
        refiner_output = None
    
    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================
    print("\n" + "="*70)
    print("✅ FINAL OUTPUT - Complete Pipeline")
    print("="*70)
    
    if scorer_output and refiner_output:
        # Build the complete question with acknowledgment
        acknowledgment = scorer_output.acknowledgment or "I see."
        final_question = refiner_output.refined_question
        
        complete_response = f"{acknowledgment} {final_question}"
        
        print(f"\n🎯 Complete Recruiter Response:")
        print(f'   "{complete_response}"')
        
        print(f"\n📈 Pipeline Summary:")
        print(f"   • Scorer identified: {scorer_output.answer_quality} quality, {scorer_output.engagement} engagement")
        print(f"   • Suggested action: {scorer_output.suggested_action}")
        print(f"   • Question was: {refiner_output.refinement_type}")
        print(f"   • Total improvements: {len(refiner_output.improvements)}")
    else:
        print("\n⚠️  Pipeline incomplete - check errors above")
    
    print("\n" + "="*70)
    print("🧪 Test Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_pipeline())
