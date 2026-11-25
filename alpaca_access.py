"""Read local credentials and make READ-ONLY requests to Alpaca."""

import json
import gzip
import http.client
import os
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class AlpacaError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"Alpaca returned HTTP {status}. Check credentials or data access.")


def load_credentials():
    """Never print or save plaintext credentials. Environment variables take priority."""
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if key and secret:
        return key, secret

    credential_file = Path(__file__).parent / ".alpaca-credentials.xml"
    if os.name == "nt" and credential_file.exists():
        # Windows decrypts the secure strings only for the user who saved them.
        # The child process output is captured in memory, never sent to the console.
        script = (
            "$c = Import-Clixml -LiteralPath $env:ENEL_ALPACA_CREDENTIAL_FILE; "
            "$k = [System.Net.NetworkCredential]::new('', $c.ApiKey).Password; "
            "$s = [System.Net.NetworkCredential]::new('', $c.Secret).Password; "
            "@{key=$k;secret=$s} | ConvertTo-Json -Compress"
        )
        environment = os.environ.copy()
        environment["ENEL_ALPACA_CREDENTIAL_FILE"] = str(credential_file)
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            values = json.loads(result.stdout)
            if result.returncode == 0 and values.get("key") and values.get("secret"):
                return values["key"], values["secret"]
        except (ValueError, AttributeError):
            pass
        raise ValueError("Windows could not read your saved credentials. Run connect_alpaca.ps1 again.")
    raise ValueError("Alpaca is not connected. Run connect_alpaca.ps1 to enter keys locally.")


class AlpacaReader:
    """A small HTTP reader; it has no order, payment, or account-change methods."""

    def __init__(self, key, secret):
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
                        "Accept-Encoding": "gzip"}
        self.last_request = 0.0
        self.rate_lock = threading.Lock()

    def get(self, host, path, parameters):
        # Credentials are sent only to these exact Alpaca service addresses.
        if host not in {"data.alpaca.markets", "paper-api.alpaca.markets", "api.alpaca.markets"}:
            raise ValueError("Unexpected service address.")
        url = "https://" + host + path + "?" + urllib.parse.urlencode(parameters)
        for attempt in range(5):
            # Stay below Basic's documented 200 requests/minute.
            with self.rate_lock:
                time.sleep(max(0, 0.4 - (time.monotonic() - self.last_request)))
                self.last_request = time.monotonic()
            request = urllib.request.Request(url, headers=self.headers, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    payload = response.read()
                    if response.headers.get("Content-Encoding", "").lower() == "gzip":
                        payload = gzip.decompress(payload)
                    return json.loads(payload)
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 4:
                    # Avoid displaying server payloads or authentication headers.
                    raise AlpacaError(error.code) from None
                time.sleep(min(2 ** (attempt + 1), 30))
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
                if attempt == 4:
                    raise RuntimeError("Could not reach Alpaca. Check the network connection.") from None
                time.sleep(min(2 ** attempt, 10))

    def calendar(self, start, end):
        parameters = {"start": start, "end": end, "date_type": "TRADING"}
        try:
            return self.get("paper-api.alpaca.markets", "/v2/calendar", parameters)
        except AlpacaError as error:
            if error.status not in {401, 403}:
                raise
            # Live keys use the live host; the request still only reads calendar data.
            return self.get("api.alpaca.markets", "/v2/calendar", parameters)
