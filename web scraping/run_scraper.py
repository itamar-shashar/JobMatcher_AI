import os
from dotenv import load_dotenv


# ============= USER CONFIGURATION =============
# Bright Data Credentials
load_dotenv()
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

# Companies to Scrape
COMPANIES_DIR = "adir_companies.csv" # Insert the directory to your companies.csv file
START_IDX = 0 # Set to the company index you want to start from
END_IDX = 2000 # Set to the company index you want to end the scraping
NUM_THREADS = 3
MAX_JOBS_PER_COMPANY = 50


# Timing Settings
TOTAL_RUNTIME_HOURS = 4    # How long the entire program should run
COOLDOWN_MINUTES = 2       # Wait time between scraping sessions
# ============================================

import time
import logging
from datetime import datetime, timedelta
import pandas as pd
from indeed_scraper import run_scraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('executor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_scraper_with_timeout(total_runtime_hours: float = 4, cooldown_minutes: float = 2, max_jobs_per_company: int = 50):
    """
    Run the scraper repeatedly for a specified total duration with cooldown periods.
    """

    start_time = datetime.now()
    end_time = start_time + timedelta(hours=total_runtime_hours)
    iteration = 1

    companies_df = pd.read_csv(COMPANIES_DIR)
    companies_df = companies_df.iloc[START_IDX:END_IDX+1]
    companies = companies_df['company'].tolist()

    logger.info(f"Starting executor. Will run until: {end_time}")
    logger.info(f"Total runtime: {total_runtime_hours} hours")
    logger.info(f"Cooldown between runs: {cooldown_minutes} minutes")

    while datetime.now() < end_time:
        try:
            current_time = datetime.now()
            time_remaining = end_time - current_time
            
            logger.info(f"\nStarting iteration {iteration}")
            logger.info(f"Time remaining: {time_remaining}")
            
            # Run the scraper with our parameters
            logger.info("Starting new scraper instance...")
            run_scraper(
                username=USERNAME,
                password=PASSWORD,
                companies=companies,
                max_workers=NUM_THREADS,
                max_jobs_per_company=max_jobs_per_company
            )
            
            # Check if we should continue
            if datetime.now() >= end_time:
                logger.info("Total runtime reached. Stopping executor.")
                break
            
            # Cooldown period
            logger.info(f"Scraper iteration {iteration} completed. Cooling down for {cooldown_minutes} minutes...")
            time.sleep(cooldown_minutes * 60)
            
            iteration += 1
            
        except KeyboardInterrupt:
            logger.info("\nExecutor interrupted by user. Shutting down gracefully...")
            break
        except Exception as e:
            logger.error(f"Error in executor: {str(e)}")
            logger.info("Waiting for cooldown period before retry...")
            time.sleep(cooldown_minutes * 60)

    total_runtime = datetime.now() - start_time
    logger.info(f"\nExecutor completed after {total_runtime.total_seconds() / 3600:.2f} hours")
    logger.info(f"Total iterations completed: {iteration - 1}")

if __name__ == "__main__":
    # Run the executor with 4 hours total runtime and 2 minutes cooldown
    run_scraper_with_timeout(TOTAL_RUNTIME_HOURS, COOLDOWN_MINUTES, MAX_JOBS_PER_COMPANY)
