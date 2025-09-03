#!/usr/bin/env python3
"""
Certification Evaluation Script

This script evaluates the RAG assistant against the certification criteria
and generates a comprehensive report.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag_chain import RAGChain
from src.vector_store import VectorStoreManager
from src.document_loader import DocumentLoader
from src.utils import create_sample_data


class CertificationEvaluator:
    """Evaluates the RAG assistant against certification criteria."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    def evaluate_document_ingestion(self):
        """Evaluate document ingestion capabilities."""
        print("📚 Evaluating Document Ingestion")
        print("=" * 40)
        
        try:
            # Create sample data
            create_sample_data("cert_test_data")
            
            # Test document loader
            loader = DocumentLoader()
            documents = loader.load_documents("cert_test_data")
            
            # Test vector store integration
            vector_store = VectorStoreManager("faiss")
            vector_store.add_documents(documents)
            
            # Evaluate
            score = 0
            if len(documents) > 0:
                score += 25
                print("✅ Document loading: 25/25")
            if vector_store.get_stats()['document_count'] > 0:
                score += 25
                print("✅ Vector store integration: 25/25")
            
            self.results['document_ingestion'] = score
            print(f"📊 Document Ingestion Score: {score}/50")
            
            # Cleanup
            import shutil
            shutil.rmtree("cert_test_data", ignore_errors=True)
            
            return score
            
        except Exception as e:
            print(f"❌ Document ingestion evaluation failed: {e}")
            self.results['document_ingestion'] = 0
            return 0
    
    def evaluate_rag_pipeline(self):
        """Evaluate the RAG pipeline functionality."""
        print("\n🔗 Evaluating RAG Pipeline")
        print("=" * 40)
        
        try:
            # Create test setup
            create_sample_data("rag_test_data")
            loader = DocumentLoader()
            documents = loader.load_documents("rag_test_data")
            
            vector_store = VectorStoreManager("faiss")
            vector_store.add_documents(documents)
            
            # Test RAG chain
            rag = RAGChain(vector_store, use_local_model=True)
            
            # Test queries
            test_queries = [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "What are the benefits of AI?",
                "What is deep learning?",
                "How can AI be applied in business?"
            ]
            
            score = 0
            successful_queries = 0
            
            for query in test_queries:
                try:
                    response = rag.query(query)
                    if response.answer and len(response.answer) > 10:
                        successful_queries += 1
                        print(f"✅ Query successful: {query[:30]}...")
                except Exception as e:
                    print(f"❌ Query failed: {query[:30]}... - {e}")
            
            # Calculate score
            if successful_queries == len(test_queries):
                score += 50
            elif successful_queries > 0:
                score += (successful_queries / len(test_queries)) * 50
            
            self.results['rag_pipeline'] = score
            print(f"📊 RAG Pipeline Score: {score}/50")
            
            # Cleanup
            import shutil
            shutil.rmtree("rag_test_data", ignore_errors=True)
            
            return score
            
        except Exception as e:
            print(f"❌ RAG pipeline evaluation failed: {e}")
            self.results['rag_pipeline'] = 0
            return 0
    
    def evaluate_user_interface(self):
        """Evaluate user interface capabilities."""
        print("\n🖥️ Evaluating User Interface")
        print("=" * 40)
        
        score = 0
        
        # Check if Streamlit app exists
        if os.path.exists("app.py"):
            score += 25
            print("✅ Streamlit interface: 25/25")
        
        # Check if CLI exists
        if os.path.exists("cli.py"):
            score += 25
            print("✅ CLI interface: 25/25")
        
        self.results['user_interface'] = score
        print(f"📊 User Interface Score: {score}/50")
        
        return score
    
    def evaluate_code_quality(self):
        """Evaluate code quality and structure."""
        print("\n🔍 Evaluating Code Quality")
        print("=" * 40)
        
        score = 0
        
        # Check project structure
        required_files = [
            "src/__init__.py",
            "src/rag_chain.py",
            "src/vector_store.py",
            "src/document_loader.py",
            "requirements.txt",
            "README.md"
        ]
        
        existing_files = sum(1 for f in required_files if os.path.exists(f))
        structure_score = (existing_files / len(required_files)) * 25
        score += structure_score
        print(f"✅ Project structure: {structure_score:.1f}/25")
        
        # Check documentation
        if os.path.exists("README.md"):
            with open("README.md", 'r') as f:
                content = f.read()
                if len(content) > 1000:  # Substantial documentation
                    score += 25
                    print("✅ Documentation: 25/25")
                else:
                    score += 10
                    print("⚠️ Basic documentation: 10/25")
        
        self.results['code_quality'] = score
        print(f"📊 Code Quality Score: {score}/50")
        
        return score
    
    def generate_report(self):
        """Generate comprehensive evaluation report."""
        print("\n" + "="*60)
        print("🎓 CERTIFICATION EVALUATION REPORT")
        print("="*60)
        
        total_score = sum(self.results.values())
        max_score = 200
        
        print(f"\n📊 OVERALL SCORE: {total_score}/{max_score} ({total_score/max_score*100:.1f}%)")
        
        print("\n📋 DETAILED RESULTS:")
        for category, score in self.results.items():
            percentage = score/50*100
            print(f"  {category.replace('_', ' ').title()}: {score}/50 ({percentage:.1f}%)")
        
        print(f"\n⏱️ Evaluation completed in {time.time() - self.start_time:.2f} seconds")
        
        # Certification status
        if total_score >= 140:  # 70% threshold
            print("\n🎉 CERTIFICATION STATUS: PASSED ✅")
            print("Your RAG assistant meets the certification requirements!")
        else:
            print("\n❌ CERTIFICATION STATUS: NEEDS IMPROVEMENT")
            print("Please address the areas with low scores.")
        
        return total_score, max_score


def main():
    """Run the certification evaluation."""
    evaluator = CertificationEvaluator()
    
    print("🚀 Starting RAG Assistant Certification Evaluation")
    print("=" * 60)
    
    # Run evaluations
    evaluator.evaluate_document_ingestion()
    evaluator.evaluate_rag_pipeline()
    evaluator.evaluate_user_interface()
    evaluator.evaluate_code_quality()
    
    # Generate report
    evaluator.generate_report()


if __name__ == "__main__":
    main()
