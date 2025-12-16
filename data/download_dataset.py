"""
NSL-KDD Dataset Download Script

Downloads the NSL-KDD dataset for network intrusion detection.
"""

import os
import urllib.request
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_file(url: str, destination: str) -> None:
    """
    Download a file from URL to destination path.
    
    Args:
        url: Source URL
        destination: Local file path to save
    """
    try:
        logger.info(f"Downloading {url}...")
        urllib.request.urlretrieve(url, destination)
        logger.info(f"Successfully downloaded to {destination}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise


def download_nsl_kdd() -> None:
    """
    Download NSL-KDD dataset files.
    
    Downloads:
        - KDDTrain+.txt (training set)
        - KDDTest+.txt (test set)
    """
    # Base URL for NSL-KDD dataset
    base_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
    
    # Create data directory
    data_dir = Path(__file__).parent
    data_dir.mkdir(exist_ok=True)
    
    # Files to download
    files = {
        "KDDTrain+.txt": f"{base_url}KDDTrain%2B.txt",
        "KDDTest+.txt": f"{base_url}KDDTest%2B.txt",
    }
    
    logger.info("Starting NSL-KDD dataset download...")
    
    for filename, url in files.items():
        destination = data_dir / filename
        
        # Skip if already exists
        if destination.exists():
            logger.info(f"{filename} already exists, skipping download")
            continue
        
        try:
            download_file(url, str(destination))
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            logger.info("\nAlternative download instructions:")
            logger.info("1. Visit: https://www.unb.ca/cic/datasets/nsl.html")
            logger.info("2. Or Kaggle: https://www.kaggle.com/datasets/hassan06/nslkdd")
            logger.info(f"3. Place files in: {data_dir}")
            return
    
    logger.info("\n[SUCCESS] Dataset download complete!")
    logger.info(f"Files saved to: {data_dir}")
    
    # Verify files
    for filename in files.keys():
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"  - {filename}: {size_mb:.2f} MB")


if __name__ == "__main__":
    download_nsl_kdd()
