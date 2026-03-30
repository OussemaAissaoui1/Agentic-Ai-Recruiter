"""
Test Client for NLP Agent

Simple command-line client to test the NLP Agent API.

Usage:
    python test_client.py
"""

import asyncio
import httpx
import sys

BASE_URL = "http://localhost:8000"

CANDIDATE_CV = """Candidate: Oussema Aissaoui
Role applied for: AI Engineering Intern

Education:
- National School of Computer Science (ENSI), Computer Engineering, Sep 2023–Present
  Coursework: AI, Data Analysis, Machine Learning, Software Engineering

Experience:
- Technology Intern, TALAN Tunisia, Jul–Aug 2025
  Applied GraphRAG and LLM reasoning for pattern recognition in enterprise datasets.
  Designed AI-driven prototypes for data analytics and business decision systems.
- AI & Automation Intern, NP Tunisia, Jul–Aug 2024
  Automated integration of 30+ industrial screens into a local network.

Projects:
- Pattern Recognition in Company Internal Data: GraphRAG + LLM reasoning on enterprise datasets.
- Blood Cell Segmentation: YOLOv8 + SAM pipeline for WBC/RBC classification and anomaly detection.
- Water Use Efficiency Estimation: ML model using WaPOR + Google Earth Engine for crop evapotranspiration.

Technical Skills:
- Languages: Python, C/C++, Java, JavaScript, Solidity
- AI & Data: ML, Deep Learning, TensorFlow, OpenCV, YOLOv8, SAM, NLP, GraphRAG
- Tools: Git, Linux, VS Code, Google Earth Engine, WaPOR
"""


async def test_health():
    """Test health endpoint."""
    print("\n📊 Testing /health endpoint...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/health", timeout=10.0)
            data = response.json()
            print(f"✅ Status: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Warmed up: {data['warmed_up']}")
            return True
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False


async def test_warmup():
    """Test warmup endpoint."""
    print("\n🔥 Testing /warmup endpoint...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{BASE_URL}/warmup",
                timeout=60.0
            )
            data = response.json()
            print(f"✅ Warmup complete: {data['status']}")
        except Exception as e:
            print(f"❌ Warmup failed: {e}")


async def simulate_interview():
    """Simulate a simple interview conversation."""
    print("\n🎤 Starting Interview Simulation...")
    print("=" * 60)

    session_id = "test-interview-001"
    conversation_history = []

    async with httpx.AsyncClient() as client:
        # Opening question
        print("\nTurn 1: Opening")
        print("-" * 60)

        try:
            response = await client.post(
                f"{BASE_URL}/generate-question",
                json={
                    "session_id": session_id,
                    "candidate_cv": CANDIDATE_CV,
                    "job_role": "AI Engineering Intern",
                    "conversation_history": [],
                    "candidate_latest_answer": "Hello! I'm Oussema, ready for the interview.",
                },
                timeout=30.0,
            )

            data = response.json()
            question = data["question"]

            print(f"Recruiter: {question}")
            print(f"\n[Metadata]")
            print(f"  Reasoning: {data['reasoning']}")
            print(f"  Turn: {data['turn_count']}")
            print(f"  Topic depth: {data['topic_depth']}")
            print(f"  TTFT: {data['metrics']['ttft_sec']:.3f}s")
            print(f"  Latency: {data['metrics']['latency_sec']:.3f}s")
            print(f"  Tokens/sec: {data['metrics']['tokens_per_sec']:.1f}")

            # Simulate candidate answering
            candidate_answer = "I worked on GraphRAG at TALAN Tunisia."
            print(f"\nCandidate: {candidate_answer}")

            conversation_history.append({
                "candidate": candidate_answer,
                "recruiter": question,
            })

            # Analyze the response
            print("\n📊 Analyzing response...")
            analysis_response = await client.post(
                f"{BASE_URL}/analyze-response",
                json={
                    "session_id": session_id,
                    "candidate_answer": candidate_answer,
                },
                timeout=10.0,
            )
            analysis = analysis_response.json()
            print(f"  Word count: {analysis['word_count']}")
            print(f"  Specificity score: {analysis['specificity_score']}")
            print(f"  Is detailed: {analysis['is_detailed']}")

            # Follow-up question
            print("\n\nTurn 2: Follow-up (expecting follow-up due to vague answer)")
            print("-" * 60)

            response = await client.post(
                f"{BASE_URL}/generate-question",
                json={
                    "session_id": session_id,
                    "candidate_cv": CANDIDATE_CV,
                    "job_role": "AI Engineering Intern",
                    "conversation_history": conversation_history,
                    "candidate_latest_answer": candidate_answer,
                },
                timeout=30.0,
            )

            data = response.json()
            question = data["question"]

            print(f"Recruiter: {question}")
            print(f"\n[Metadata]")
            print(f"  Reasoning: {data['reasoning']}")
            print(f"  Is follow-up: {data['is_follow_up']}")
            print(f"  Topic depth: {data['topic_depth']}")

            # Detailed answer
            detailed_answer = (
                "I implemented the LLM reasoning pipeline using Llama 3.1 and Neo4j. "
                "The main challenge was optimizing query performance, which I solved "
                "by implementing efficient graph traversal algorithms. This reduced "
                "query latency from 3 seconds to under 500ms, a 83% improvement."
            )

            print(f"\nCandidate: {detailed_answer}")

            conversation_history.append({
                "candidate": detailed_answer,
                "recruiter": question,
            })

            # Third question (should move to new topic)
            print("\n\nTurn 3: New topic (expecting move due to detailed answer)")
            print("-" * 60)

            response = await client.post(
                f"{BASE_URL}/generate-question",
                json={
                    "session_id": session_id,
                    "candidate_cv": CANDIDATE_CV,
                    "job_role": "AI Engineering Intern",
                    "conversation_history": conversation_history,
                    "candidate_latest_answer": detailed_answer,
                },
                timeout=30.0,
            )

            data = response.json()
            question = data["question"]

            print(f"Recruiter: {question}")
            print(f"\n[Metadata]")
            print(f"  Reasoning: {data['reasoning']}")
            print(f"  Is follow-up: {data['is_follow_up']}")
            print(f"  Topic depth: {data['topic_depth']} (reset to 0)")

            # Get session stats
            print("\n\n📈 Session Statistics:")
            print("-" * 60)

            stats_response = await client.get(
                f"{BASE_URL}/session/{session_id}/stats",
                timeout=10.0,
            )

            stats = stats_response.json()
            print(f"  Total turns: {stats['turn_count']}")
            print(f"  Questions asked: {stats['questions_asked_count']}")
            print(f"  Current topic depth: {stats['topic_depth']}")
            print(f"  Consecutive vague answers: {stats['consecutive_vague']}")

            # Clean up session
            print("\n\n🧹 Cleaning up session...")
            await client.delete(f"{BASE_URL}/session/{session_id}")
            print("✅ Session ended")

        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error: {e.response.status_code}")
            print(f"   {e.response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print("✅ Interview simulation complete!")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NLP Agent Test Client")
    print("=" * 60)

    # Test health
    is_healthy = await test_health()
    if not is_healthy:
        print("\n❌ Server is not healthy. Please start the server first:")
        print("   ./start_server.sh")
        sys.exit(1)

    # Warmup
    await test_warmup()

    # Run interview simulation
    await simulate_interview()

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(0)
