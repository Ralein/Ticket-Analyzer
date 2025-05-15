import requests
import sqlite3
import json
from datetime import datetime
import asyncio
import aiohttp
import os

class TicketAnalyzer:
    DEFAULT_API_URLS = [""]
    AVAILABLE_MODELS = [
        "deepseek-r1:1.5b",
        "mistral:latest",
        "openchat:latest",
        "llama2:latest",
        "deepseek-r1:7b"
    ]
    SIMILARITY_THRESHOLD = 0.7
    MAX_TOKENS = 100
    TEMPERATURE = 0.2

    def __init__(self, api_url=None, model=None):
        self.api_url = self._find_working_api_url(api_url)
        self.db_path = "tickets.db"
        self.DEFAULT_MODEL = model if model else "mistral:latest"
        self.setup_database()

    def _find_working_api_url(self, api_url):
        urls = [api_url] if api_url else self.DEFAULT_API_URLS
        for url in urls:
            try:
                response = requests.get(url, timeout=3)
                response.raise_for_status()
                print(f"Connected to Ollama server at {url}")
                return url
            except Exception:
                print(f"⚠️ Failed to connect to {url}")
        print("⚠️ No Ollama server available. Functionality will be limited.")
        return urls[-1]  # Default to localhost for error messages

    def setup_database(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY,
                    description TEXT,
                    model TEXT,
                    categories TEXT,
                    solution TEXT,
                    timestamp DATETIME,
                    similarity_group INTEGER
                )
            ''')
            cursor.execute('PRAGMA table_info(tickets)')
            columns = [column[1] for column in cursor.fetchall()]
            if 'solution' not in columns:
                cursor.execute('ALTER TABLE tickets ADD COLUMN solution TEXT')
                conn.commit()
            conn.commit()
        except sqlite3.DatabaseError as e:
            print(f"⚠️ Database error: {e}")
            print("⚠️ Database is corrupted. Deleting and recreating a new one.")
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            try:
                os.remove(self.db_path)
            except Exception as e:
                print(f"Could not remove corrupted database: {e}")
            self.setup_database()
        finally:
            if conn:
                conn.close()

    def get_solution(self, ticket_id, description):
        prompt = (
            f"You are a support agent. Given the following ticket, determine if it is a repeat of a previous issue. "
            f"If yes, provide the old ticket ID and a brief solution. If not, generate a concise solution for the new ticket.\n\n"
            f"Ticket ID: {ticket_id}\nDescription: {description}\n\n"
            "Respond in the format:\n"
            "REPEAT: <yes/no>\nOLD_TICKET_ID: <id if repeat, else N/A>\nSOLUTION: <solution>"
        )
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={
                    "model": self.DEFAULT_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "max_tokens": self.MAX_TOKENS,
                    "temperature": self.TEMPERATURE
                },
                timeout=60
            )
            response.raise_for_status()
            response_data = response.json()
            if not response_data.get('response'):
                return "Error: Empty response from API"
            # Extract solution
            lines = response_data['response'].splitlines()
            for line in lines:
                if line.startswith("SOLUTION:"):
                    return line.replace("SOLUTION:", "").strip()
            return response_data['response'].strip()
        except requests.exceptions.Timeout:
            return "Error: Request timed out. Please check if Ollama server is running."
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama server. Please verify the server is running and accessible."
        except Exception as e:
            return f"Error getting solution: {str(e)}"

    async def fetch_similarity(self, session, new_description, ticket_id, existing_description):
        prompt = (
            f"You are a support agent. Compare the following two tickets and determine their similarity (0 to 1). "
            f"If they are similar (similarity > {self.SIMILARITY_THRESHOLD}), provide the similarity score and the solution for the existing ticket. "
            f"Otherwise, just provide the similarity score.\n\n"
            f"New Ticket: {new_description}\n"
            f"Existing Ticket (ID: {ticket_id}): {existing_description}\n\n"
            "Respond in the format:\n"
            "SIMILARITY: <score>\nSOLUTION: <solution if similar, else N/A>"
        )
        max_retries = 3
        timeout = 60
        for attempt in range(max_retries):
            try:
                response = await session.post(
                    f"{self.api_url}/api/generate",
                    json={
                        "model": self.DEFAULT_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "max_tokens": self.MAX_TOKENS,
                        "temperature": self.TEMPERATURE
                    },
                    timeout=timeout
                )
                response.raise_for_status()
                return ticket_id, existing_description, await response.json()
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    print(f"Timeout on attempt {attempt + 1} for ticket {ticket_id}, retrying...")
                    await asyncio.sleep(1)
                else:
                    print(f"Failed after {max_retries} attempts for ticket {ticket_id}")
                    return ticket_id, existing_description, None
            except Exception as e:
                print(f"Error fetching similarity for ticket {ticket_id} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    return ticket_id, existing_description, None

    async def find_similar_tickets(self, new_description):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, description FROM tickets")
            tickets = cursor.fetchall()
        finally:
            conn.close()

        if not tickets:
            print("No tickets found in database")
            return []

        similar_tickets = []
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.fetch_similarity(session, new_description, ticket_id, description)
                for ticket_id, description in tickets
            ]
            results = await asyncio.gather(*tasks)

            for ticket_id, existing_description, result in results:
                if result:
                    response_text = result.get("response", "").strip()
                    similarity_score = None
                    solution = None
                    for line in response_text.splitlines():
                        if line.startswith("SIMILARITY:"):
                            try:
                                similarity_score = float(line.replace("SIMILARITY:", "").strip())
                            except Exception:
                                similarity_score = None
                        if line.startswith("SOLUTION:"):
                            solution = line.replace("SOLUTION:", "").strip()
                    if similarity_score is not None and similarity_score > self.SIMILARITY_THRESHOLD:
                        similar_tickets.append({
                            "id": ticket_id,
                            "description": existing_description,
                            "solution": solution,
                            "similarity": similarity_score
                        })

        return similar_tickets

    def check_ticket_similarity(self, ticket_id, description):
        similar_tickets = asyncio.run(self.find_similar_tickets(description))
        if similar_tickets:
            print(f"\nSimilar tickets to '{ticket_id}':")
            for ticket in similar_tickets:
                print(f"Ticket ID: {ticket['id']} (Similarity: {ticket['similarity']:.2f})")
                print(f"Description: {ticket['description']}")
                print(f"Suggested Solution: {ticket['solution']}\n")
        else:
            print("No similar tickets found.")

    def add_ticket(self, ticket_id, description, category):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
            if cursor.fetchone():
                print(f"Error: Ticket {ticket_id} already exists")
                return False

            similar_tickets = asyncio.run(self.find_similar_tickets(description))
            if similar_tickets:
                solution = similar_tickets[0]['solution'] or "No solution found"
            else:
                solution = self.get_solution(ticket_id, description)

            cursor.execute('''
                INSERT INTO tickets (id, description, model, categories, solution, timestamp, similarity_group)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticket_id,
                description,
                self.DEFAULT_MODEL,
                category,  # Store category as a string
                solution,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                1
            ))

            conn.commit()
            print(f"Ticket '{ticket_id}' added at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Description: {description}")
            print(f"Category: {category}")
            print(f"Generated Solution: {solution}")
            return True
        except Exception as e:
            print(f"Error adding ticket: {str(e)}")
            return False
        finally:
            conn.close()

    def get_all_tickets(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, description, solution, model, categories, timestamp, similarity_group FROM tickets")
            tickets = cursor.fetchall()
            if not tickets:
                print("No tickets found in database")
                return
        except Exception as e:
            print(f"Error retrieving tickets: {e}")
            return
        finally:
            if conn:
                conn.close()

        print("\nAll Tickets:")
        print("=" * 100)
        header = f"| {'ID':<15} | {'Description':<50} | {'Model':<15} | {'Category':<15} | {'Solution':<30} | {'Timestamp':<20} | {'Similarity Group':<15} |"
        print(header)
        print("=" * len(header))
        for ticket in tickets:
            tid = ticket[0][:15] if ticket[0] else ""
            desc = ticket[1][:50] if ticket[1] else ""
            model = ticket[3][:15] if ticket[3] else ""
            categories = ticket[4][:15] if ticket[4] else ""
            solution = ticket[2][:30] if ticket[2] else "No solution"
            timestamp = ticket[5]
            similarity_group = ticket[6]
            print(f"| {tid:<15} | {desc:<50} | {model:<15} | {categories:<15} | {solution:<30} | {timestamp:<20} | {similarity_group:<15} |")
        print("=" * 100)

    def get_category(self, ticket_id, description):
        categories_list = (
            "Hardware, Software, Network, Email, Login/Access, Peripheral, Performance, Security, Printer, Database, "
            "Server, Cloud Services, Backup/Recovery, Application Support, UI/UX Bug, Mobile Device, Internet Connectivity, "
            "Power Supply, Operating System, File System, Permissions, Malware/Virus, Phishing, Patch Management, VPN, DNS, "
            "Firewall, Proxy, Storage, Device Configuration, Account Management, Password Reset, Multi-Factor Authentication, "
            "Licensing, Deployment, Installation, Update Issues, System Crash, Blue Screen, Boot Failure, Slow Boot, "
            "Application Crash, Data Loss, Sync Issues, Drive Mapping, Network Drive, Printer Driver, Monitor Display, "
            "Keyboard/Mouse Issue, Touchscreen Problem, Battery Issue, BIOS/UEFI, Disk Errors, Network Cable Issue, "
            "Wi-Fi Signal Issue, Port Blocked, Email Spam, Email Configuration, Mailbox Full, Shared Folder Issue, "
            "Remote Desktop, Software Compatibility, Software Activation, Virtual Machine Issue, Cloud Sync, SaaS Access, "
            "Hosting Issue, SSL Certificate, Configuration Error, Scheduler Issue, Cron Job, API Error, Integration Issue, "
            "Data Migration, Report Generation, Analytics Error, Dashboard Issue, Backup Disk Full, Restore Failure, "
            "Access Denied, User Provisioning, System Freeze, Thermal Overheating, Resource Exhaustion, Kernel Panic, "
            "Registry Error, Log File Issue, Proxy Authentication, Wireless Adapter, IP Conflict, Bandwidth Throttling, "
            "QoS Issue, Cloud Storage Quota, Cloud Login Issue, Mobile App Crash, Mobile Sync Issue, UX Feedback, "
            "Documentation Error, Unsupported Feature, Version Mismatch, Plugin Error, Browser Compatibility, Cross-Site Error, "
            "Database Timeout, Connection Pool Error, Latency Issue, Packet Loss, Audit Log, Encryption Problem, "
            "Certificate Expired, Compliance Violation"
        )
        categories = [c.strip() for c in categories_list.split(",")]
        prompt = (
            f"You are an expert IT support agent. Given the ticket description below, select the single most appropriate category from the provided list. "
            f"Choose only one category and do not invent new categories. Respond only with the format:\nCATEGORY: <category from the list below>\n\n"
            f"Categories:\n{categories_list}\n\n"
            f"Ticket ID: {ticket_id}\nDescription: {description}\n"
        )
        max_retries = 3
        timeout = 60
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_url}/api/generate",
                    json={
                        "model": self.DEFAULT_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "max_tokens": self.MAX_TOKENS,
                        "temperature": self.TEMPERATURE
                    },
                    timeout=timeout
                )
                response.raise_for_status()
                response_data = response.json()
                if not response_data.get('response'):
                    print("AI response was empty.")
                    return "Uncategorized"
                category = "Uncategorized"
                lines = response_data['response'].splitlines()
                for line in lines:
                    if line.strip().lower().startswith("category:"):
                        category_candidate = line.split(":", 1)[1].strip()
                        # Case-insensitive match to handle AI output variations
                        for valid in categories:
                            if category_candidate.lower() == valid.lower():
                                category = valid
                                break
                        else:
                            print(f"AI returned unknown category: '{category_candidate}'")
                            category = "Uncategorized"
                        break
                else:
                    print(f"No CATEGORY line found in AI response: {response_data['response']}")
                return category
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1} for ticket {ticket_id}, retrying...")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5)
                else:
                    print("Failed to categorize ticket due to repeated timeouts.")
                    return "Uncategorized"
            except requests.exceptions.ConnectionError:
                print("Could not connect to Ollama server. Please verify the server is running and accessible.")
                return "Uncategorized"
            except Exception as e:
                print(f"Error getting category: {str(e)}")
                return "Uncategorized"

if __name__ == "__main__":
    print("Available models:")
    for idx, model in enumerate(TicketAnalyzer.AVAILABLE_MODELS, 1):
        print(f"{idx}. {model}")
    while True:
        model_choice = input("Select the model to use (enter number): ").strip()
        try:
            model_idx = int(model_choice) - 1
            if 0 <= model_idx < len(TicketAnalyzer.AVAILABLE_MODELS):
                selected_model = TicketAnalyzer.AVAILABLE_MODELS[model_idx]
                break
            else:
                print("Invalid selection. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    ai = TicketAnalyzer(model=selected_model)
    while True:
        ticket_id = input("Enter Ticket ID: ").strip()
        if not ticket_id:
            print("Error: Ticket ID cannot be empty")
            continue
        break
    while True:
        description = input("Enter Ticket Description: ").strip()
        if not description:
            print("Error: Description cannot be empty")
            continue
        break
    # Use AI to categorize based on description
    category = ai.get_category(ticket_id, description)
    print(f"Category: {category}")
    ai.check_ticket_similarity(ticket_id, description)
    if ai.add_ticket(ticket_id, description, category):
        ai.get_all_tickets()
