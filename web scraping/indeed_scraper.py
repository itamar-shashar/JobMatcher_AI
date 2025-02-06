import time , os
import random
import re
import urllib.parse
import json
import concurrent.futures
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from queue import Queue
from threading import Lock
import logging

from selenium import webdriver
from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class JobListing:
    """Data class to store job listing information."""
    company_name: str
    title: str
    url: str
    location: str
    salary: Optional[str] = None
    job_type: Optional[str] = None
    description: Optional[str] = None


class ProxyManager:
    """Manages proxy connections and handles cooling periods."""

    def __init__(self, username: str, password: str):
        self.auth = f"{username}:{password}"
        self.base_url = "https://zproxy.lum-superproxy.io:9515"
        self.cooling_period = 10
        self.max_retries = 3

    @property
    def proxy_url(self) -> str:
        return f"https://{self.auth}@zproxy.lum-superproxy.io:9515"

    def handle_request(self, func, *args, **kwargs):
        """Execute a request with retry logic and cooling period handling."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if 'proxy_no_peers_cooling' in str(e) or '502' in str(e):
                    cooling_time = self.cooling_period * (1.5 ** attempt) + random.uniform(1,
                                                                                           3)  # Reduced exponential factor
                    logger.warning(
                        f"Proxy cooling triggered. Waiting {cooling_time:.1f}s (Attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(cooling_time)
                    if attempt == self.max_retries - 1:
                        raise Exception("Max retries reached for proxy cooling")
                    continue
                raise e
        return None


class IndeedScraper:
    """Parallel Indeed job listing scraper with proxy support."""

    def __init__(
            self,
            username: str,
            password: str,
            max_workers: int = 3,
            time_limit_minutes: int = 27,  # New parameter
            jobs_per_company_defualt: int = 10,
            save_interval: int = 5,
            max_jobs_per_company: int = 50
    ):
        self.proxy_manager = ProxyManager(username, password)
        self.max_workers = max_workers
        self.time_limit_minutes = time_limit_minutes
        self.jobs_per_company_defualt = jobs_per_company_defualt
        self.save_interval = save_interval
        self.max_jobs_per_company = max_jobs_per_company

        self.collected_jobs: List[JobListing] = []
        self.companies_processed: List[str] = []
        self.jobs_lock = Lock()
        self.save_lock = Lock()
        self.url_lock = Lock()
        self.processed_urls = set()
        self.start_time = None

    def _check_time_limit(self) -> bool:
        """Check if we're approaching the time limit."""
        if not self.start_time:
            return False
        elapsed_minutes = (time.time() - self.start_time) / 60
        return elapsed_minutes >= self.time_limit_minutes

    def _save_company_progress(self, company: str, page: int) -> None:
        """Save progress for a specific company."""
        with self.save_lock:
            try:
                try:
                    with open('scraping_progress.json', 'r') as f:
                        progress = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    progress = {
                        'companies_processed': self.companies_processed,
                        'total_jobs': len(self.collected_jobs),
                        'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'company_progress': {}
                    }

                # Update company-specific progress
                if 'company_progress' not in progress:
                    progress['company_progress'] = {}
                progress['company_progress'][company] = {
                    'last_page': page,
                    'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
                }

                with open('scraping_progress.json', 'w') as f:
                    json.dump(progress, f, indent=4)

            except Exception as e:
                logger.error(f"Error saving company progress: {str(e)}")

    def _init_driver(self) -> Remote:
        """Initialize and return a Selenium webdriver with proxy settings."""
        connection = ChromiumRemoteConnection(self.proxy_manager.proxy_url, 'goog', 'chrome')
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        return Remote(connection, options=options)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_job_details(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract detailed job information from the job page."""
        details = {'salary': None, 'job_type': None, 'description': None}

        try:
            # Extract salary and job type
            other_details = soup.find('div', attrs={'data-testid': 'jobsearch-OtherJobDetailsContainer'})
            if other_details:
                salary_type_div = other_details.find('div', {'id': 'salaryInfoAndJobType'})
                if salary_type_div:
                    spans = salary_type_div.find_all('span')
                    for span in spans:
                        text = span.text.strip()
                        if '$' in text:
                            details['salary'] = text
                        elif text and '$' not in text:
                            details['job_type'] = text

            # Extract description
            description_div = soup.find('div', {'id': 'jobDescriptionText'})
            if description_div:
                details['description'] = self._clean_text(description_div.text)

        except Exception as e:
            logger.error(f"Error extracting job details: {str(e)}")

        return details

    def _process_job_listing(self, job_data: dict, driver: Remote) -> Optional[JobListing]:
        """Process a single job listing."""
        try:
            # Get job details page
            self.proxy_manager.handle_request(
                driver.get,
                job_data['url']
            )
            time.sleep(random.uniform(7, 9))  # Reduced from 8-10

            if 'captcha' in driver.current_url.lower():
                logger.warning("Captcha detected, waiting...")
                time.sleep(random.uniform(12, 15))  # Reduced from 20-25

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            details = self._extract_job_details(soup)

            return JobListing(
                company_name=job_data['company_name'],
                title=job_data['title'],
                url=job_data['url'],
                location=job_data['location'],
                salary=details['salary'],
                job_type=details['job_type'],
                description=details['description']
            )

        except Exception as e:
            logger.error(f"Error processing job {job_data['url']}: {str(e)}")
            return None

    def _worker(self, job_queue: Queue, pbar: tqdm) -> List[JobListing]:
        """Worker function for processing jobs in parallel."""
        collected_jobs = []
        with self._init_driver() as driver:
            while True:
                try:
                    job_data = job_queue.get_nowait()
                except:
                    break

                job = self._process_job_listing(job_data, driver)
                if job:
                    collected_jobs.append(job)
                job_queue.task_done()
                pbar.update(1)

        return collected_jobs

    def _save_progress(self) -> None:
        """Save current progress to file."""
        with self.save_lock:
            try:
                temp_file = 'job_listings_tmp.pkl'
                final_file = 'job_listings.pkl'

                # Read existing data if the pickle file exists
                if os.path.exists(final_file):
                    try:
                        existing_df = pd.read_pickle(final_file, compression='gzip')
                    except Exception as e:
                        logger.error(f"Error reading existing pickle file: {str(e)}")
                        existing_df = pd.DataFrame()
                else:
                    existing_df = pd.DataFrame()

                # Combine with new data
                new_df = pd.DataFrame([vars(job) for job in self.collected_jobs])
                combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates()

                # Write the combined data to a temporary file and then rename it atomically
                combined_df.to_pickle(temp_file, compression='gzip')
                os.replace(temp_file, final_file)

                # # Save progress metadata
                # progress = {
                #     'companies_processed': self.companies_processed,
                #     'total_jobs': len(combined_df),
                #     'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
                # }
                #
                # with open('scraping_progress_tmp.json', 'w') as f:
                #     json.dump(progress, f, indent=4)
                # os.replace('scraping_progress_tmp.json', 'scraping_progress.json')

                logger.info(f"Progress saved: {len(combined_df)} jobs from {len(self.companies_processed)} companies")
            except Exception as e:
                logger.error(f"Error saving progress: {str(e)}")

    def scrape_company(self, company: str) -> List[JobListing]:
        """Scrape job listings for a single company."""
        logger.info(f"Processing company: {company}")
        collected_job_urls = set()
        collected_jobs = []

        try:
            # Load progress for this company if it exists
            last_processed_page = 0
            try:
                with open('scraping_progress.json', 'r') as f:
                    progress = json.load(f)
                    if 'company_progress' in progress and company in progress['company_progress']:
                        last_processed_page = progress['company_progress'][company]['last_page']
                        logger.info(f"Resuming {company} from page {last_processed_page}")
            except (FileNotFoundError, json.JSONDecodeError):
                pass

            with self._init_driver() as driver:
                page = last_processed_page
                total_jobs_target = self.max_jobs_per_company

                while True:
                    # Check time limit before processing each page
                    if self._check_time_limit():
                        logger.warning(f"Time limit approaching for {company}, saving progress...")
                        self._save_company_progress(company, page)
                        return collected_jobs

                    # Add shorter delay between pages
                    if page > 0:
                        cooling_time = random.uniform(15, 20)  # Reduced from 25-35
                        logger.info(f"Cooling between pages for {cooling_time:.1f} seconds")
                        time.sleep(cooling_time)

                    # Construct URL with pagination
                    start_param = f"&start={page * 10}" if page > 0 else ""
                    search_url = f"https://www.indeed.com/jobs?q={urllib.parse.quote_plus(company)}{start_param}"

                    # Use multiple retries for page loading with shorter delays
                    max_page_retries = 3
                    for retry in range(max_page_retries):
                        try:
                            self.proxy_manager.handle_request(driver.get, search_url)
                            time.sleep(random.uniform(10, 15))  # Reduced from 15-20

                            # Additional check for page load
                            if 'indeed.com' not in driver.current_url:
                                raise Exception("Page not properly loaded")

                            break
                        except Exception as e:
                            if retry == max_page_retries - 1:
                                raise e
                            cooling_time = random.uniform(15, 20)  # Reduced from 30-45
                            logger.warning(
                                f"Page load failed, cooling for {cooling_time:.1f}s (Attempt {retry + 1}/{max_page_retries})")
                            time.sleep(cooling_time)

                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    job_containers = soup.find_all('div', class_='job_seen_beacon')

                    # Get total number of jobs on first page
                    if page == 0:
                        try:
                            job_search_details = soup.find('div', class_='jobsearch-JobCountAndSortPane-jobCount')
                            if job_search_details:
                                results_text = job_search_details.find('span').text
                                num_of_results_found = int(''.join(filter(str.isdigit, results_text)) or 0)
                                total_jobs_target = min(num_of_results_found // 2,
                                                        self.max_jobs_per_company) if num_of_results_found > 30 else num_of_results_found
                                logger.info(f"Target jobs for {company}: {total_jobs_target}")
                        except Exception as e:
                            logger.warning(f"Could not determine total jobs for {company}: {str(e)}")
                            total_jobs_target = self.jobs_per_company_defualt

                    if not job_containers:
                        logger.warning(f"No job listings found for {company} on page {page + 1}")
                        break

                    # Process jobs for current page
                    jobs_queue = Queue()
                    page_job_count = 0

                    # Collect job URLs and basic info for current page
                    for container in job_containers:
                        try:
                            title_box = container.find('h2', class_='jobTitle')
                            link_element = title_box.find('a')
                            job_url = f"https://indeed.com/viewjob?{link_element['href'].split('?')[1]}"

                            # Thread-safe URL deduplication
                            with self.url_lock:
                                if job_url in self.processed_urls:
                                    continue
                                self.processed_urls.add(job_url)
                                collected_job_urls.add(job_url)

                            job_data = {
                                'title': title_box.find('span').text.strip(),
                                'company_name': container.find('span',
                                                               attrs={'data-testid': 'company-name'}).text.strip(),
                                'location': container.find('div',
                                                           attrs={'data-testid': 'text-location'}).text.strip(),
                                'url': job_url
                            }

                            jobs_queue.put(job_data)
                            page_job_count += 1

                        except Exception as e:
                            logger.error(f"Error collecting job info: {str(e)}")
                            continue

                    # Process jobs from current page
                    if page_job_count > 0:
                        with tqdm(total=page_job_count,
                                  desc=f"Processing {company} jobs (page {page + 1})",
                                  unit="job",
                                  leave=False) as pbar:
                            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                                future_to_jobs = [
                                    executor.submit(self._worker, jobs_queue, pbar)
                                    for _ in range(min(self.max_workers, jobs_queue.qsize()))
                                ]

                                for future in concurrent.futures.as_completed(future_to_jobs):
                                    try:
                                        jobs = future.result()
                                        collected_jobs.extend(jobs)
                                    except Exception as e:
                                        logger.error(f"Worker failed: {str(e)}")

                    # Save progress after each page
                    self._save_company_progress(company, page)

                    if self._check_time_limit():
                        logger.warning(f"Time limit reached after processing page {page} for {company}")
                        return collected_jobs

                    # Check if we've collected enough jobs
                    if total_jobs_target and len(collected_jobs) >= total_jobs_target:
                        logger.info(f"Reached target number of jobs ({total_jobs_target}) for {company}")
                        break

                    # Check if there's a next page
                    next_page = soup.find('a', {'aria-label': 'Next Page'})
                    if not next_page:
                        logger.info(f"No more pages available for {company}")
                        break

                    page += 1
                    logger.info(f"Moving to page {page + 1} for {company}")

                return collected_jobs

        except Exception as e:
            logger.error(f"Error processing company {company}: {str(e)}")
            self._save_company_progress(company, page)
            return collected_jobs

    def run2(self, companies: List[str]) -> None:
        """Run the parallel scraping process with time limit."""
        self.start_time = time.time()
        logger.info(f"Starting parallel Indeed job scraper (time limit: {self.time_limit_minutes} minutes)...")

        # Load progress to skip completed companies
        try:
            with open('scraping_progress.json', 'r') as f:
                progress = json.load(f)
                completed_companies = set(progress.get('companies_processed', []))
                companies = [c for c in companies if c not in completed_companies]
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        for company in tqdm(companies, desc="Processing companies", unit="company"):
            if self._check_time_limit():
                logger.warning("Time limit reached, stopping gracefully...")
                break

            try:
                jobs = self.scrape_company(company)

                with self.jobs_lock:
                    self.collected_jobs.extend(jobs)
                    self.companies_processed.append(company)

                if len(self.companies_processed) % self.save_interval == 0:
                    self._save_progress()

            except Exception as e:
                logger.error(f"Failed to process company {company}: {str(e)}")
                continue

        self._save_progress()

        elapsed_time = (time.time() - self.start_time) / 60
        logger.info(f"Scraping completed in {elapsed_time:.2f} minutes")
        logger.info(f"Companies processed: {len(self.companies_processed)}")
        logger.info(f"Total jobs collected: {len(self.collected_jobs)}")


    def run(self, companies: List[str]) -> None:
        """Run the parallel scraping process with time limit."""
        self.start_time = time.time()
        logger.info(f"Starting parallel Indeed job scraper (time limit: {self.time_limit_minutes} minutes)...")

        # Load progress to skip completed companies
        try:
            with open('scraping_progress.json', 'r') as f:
                progress = json.load(f)
                completed_companies = set(progress.get('companies_processed', []))
                companies = [c for c in companies if c not in completed_companies]
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # Partition companies by thread index
        partitions = {i: [] for i in range(self.max_workers)}
        for idx, company in enumerate(companies):
            partitions[idx % self.max_workers].append(company)

        # Create a worker function to process assigned companies
        def worker_function(thread_idx: int, company_subset: List[str]):
            for company in tqdm(company_subset, desc=f"Thread-{thread_idx} processing companies", unit="company"):
                if self._check_time_limit():
                    logger.warning(f"Thread-{thread_idx}: Time limit reached, stopping gracefully...")
                    return

                try:
                    jobs = self.scrape_company(company)

                    with self.jobs_lock:
                        self.collected_jobs.extend(jobs)
                        self.companies_processed.append(company)

                    if len(self.companies_processed) % self.save_interval == 0:
                        self._save_progress()

                except Exception as e:
                    logger.error(f"Thread-{thread_idx}: Failed to process company {company}: {str(e)}")

        # Run workers in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(worker_function, thread_idx, company_subset)
                for thread_idx, company_subset in partitions.items()
            ]
            concurrent.futures.wait(futures)

        self._save_progress()

        elapsed_time = (time.time() - self.start_time) / 60
        logger.info(f"Scraping completed in {elapsed_time:.2f} minutes")
        logger.info(f"Companies processed: {len(self.companies_processed)}")
        logger.info(f"Total jobs collected: {len(self.collected_jobs)}")


def run_scraper(username, password, companies, max_workers=3, time_limit_minutes=27, jobs_per_company_defualt=10,
                save_interval=5, max_jobs_per_company=50):
    scraper = IndeedScraper(
        username=username,
        password=password,
        max_workers=max_workers,
        time_limit_minutes=time_limit_minutes,
        jobs_per_company_defualt=jobs_per_company_defualt,
        save_interval=save_interval,
        max_jobs_per_company=max_jobs_per_company
    )

    scraper.run(companies)
