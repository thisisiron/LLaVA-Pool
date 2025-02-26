"""
Script for uploading HuggingFace models, configurations, and processors to the hub
"""
import argparse
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    AutoProcessor
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='HuggingFace Model Upload Script')
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to the base model to upload"
    )
    parser.add_argument(
        "--repo_path",
        type=str,
        required=True,
        help="Repository path on HuggingFace Hub"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default='',
        help="HuggingFace authentication token"
    )
    return parser.parse_args()


def main():
    """Main execution function"""
    args = get_args()
    logger.info("Starting model loading")

    # Load config, model, and processor
    config = AutoConfig.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    try:
        processor = AutoProcessor.from_pretrained(args.base_model)
        has_processor = True
        logger.info("Processor loaded successfully")
    except Exception as e:
        logger.warning(f"No processor found: {e}")
        has_processor = False

    # Prepare for HuggingFace Hub upload
    logger.info(f"Starting upload to {args.repo_path}")
    
    config.push_to_hub(
        args.repo_path,
        use_temp_dir=True,
        use_auth_token=args.hf_token
    )
    
    model.push_to_hub(
        args.repo_path,
        use_temp_dir=True,
        use_auth_token=args.hf_token
    )

    if has_processor:
        processor.push_to_hub(
            args.repo_path,
            use_temp_dir=True,
            use_auth_token=args.hf_token
        )
        logger.info("Processor uploaded successfully")

    logger.info(f"Model successfully uploaded to {args.repo_path}")


if __name__ == "__main__":
    main()
