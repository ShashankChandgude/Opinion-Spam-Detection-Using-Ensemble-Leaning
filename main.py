#!/usr/bin/env python
# coding: utf-8

import argparse
import matplotlib
matplotlib.use("Agg")

from src.utils.pipeline_orchestrator import PipelineOrchestrator
from src.utils.config import config
from src.utils.logging_config import setup_logging

def parse_arguments():
    """Parse command line arguments - Single Responsibility: CLI parsing."""
    parser = argparse.ArgumentParser(description="Opinion-Spam Detection: full train/eval pipeline")
    parser.add_argument(
        "--vectorizer",
        choices=["count", "tfidf"],
        default=config.DEFAULT_VECTORIZER,
        help="which text-vectorizer to use"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=config.TEST_SIZE,
        help="fraction of data to hold out for test"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    setup_logging()
        
    orchestrator = PipelineOrchestrator()
    results = orchestrator.run(
        vectorizer_type=args.vectorizer,
        test_size=args.test_size
    )
    
    print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
