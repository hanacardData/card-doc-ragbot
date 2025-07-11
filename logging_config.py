### logging_config.py
import logging
import logging.handlers
import os
from datetime import datetime

def setup_logging():
    # 로그 디렉토리 생성
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # 로그 파일 이름 (날짜 기반)
    log_file = os.path.join(log_dir, f'chatbot_{datetime.now().strftime("%Y%m%d")}.log')

    # 로거 설정
    logger = logging.getLogger('ChatbotLogger')
    logger.setLevel(logging.DEBUG)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 파일 핸들러 (로테이션)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)

    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 핸들러 추가
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger