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
    """커맨드 라인 인자를 파싱하는 함수"""
    parser = argparse.ArgumentParser(description='HuggingFace 모델 업로드 스크립트')
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="업로드할 기본 모델 경로"
    )
    parser.add_argument(
        "--repo_path",
        type=str,
        required=True,
        help="HuggingFace Hub의 저장소 경로"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default='',
        help="HuggingFace 인증 토큰"
    )
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = get_args()
    logger.info("모델 로딩 시작")

    # 설정, 모델, 프로세서 로딩
    config = AutoConfig.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    try:
        processor = AutoProcessor.from_pretrained(args.base_model)
        has_processor = True
        logger.info("프로세서 로딩 완료")
    except Exception as e:
        logger.warning(f"프로세서를 찾을 수 없습니다: {e}")
        has_processor = False

    # HuggingFace Hub 업로드 준비
    logger.info(f"{args.repo_path}에 업로드 시작")
    
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
        logger.info("프로세서 업로드 완료")

    logger.info(f"모델이 {args.repo_path}에 성공적으로 업로드되었습니다.")


if __name__ == "__main__":
    main()
