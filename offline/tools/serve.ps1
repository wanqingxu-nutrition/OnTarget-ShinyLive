# OnTarget - Diet Optimizer : offline launcher
#
# Serves the ./site folder over http://127.0.0.1 and opens it in the default
# browser. A web server is required because ShinyLive registers a service
# worker, and browsers refuse to register one for a page opened from the file
# system (file://). http://127.0.0.1 counts as a secure context, so it works.
#
# Uses only components that ship with Windows - no Python, no installation, no
# admin rights, no network access.

$ErrorActionPreference = "Stop"

# site/ sits next to the tools/ folder this script lives in.
$root = Split-Path -Parent $PSScriptRoot
$siteDir = Join-Path $root "site"

if (-not (Test-Path (Join-Path $siteDir "index.html"))) {
    Write-Host ""
    Write-Host "  ERROR: could not find site\index.html next to this launcher." -ForegroundColor Red
    Write-Host "  Unzip the whole folder first, keeping Start-OnTarget.bat and site\ together."
    Write-Host ""
    Read-Host "  Press Enter to close"
    exit 1
}

# Content types the app actually needs. .wasm must be exact or the browser
# refuses to stream-compile it; unknown types fall back to octet-stream.
$mime = @{
    ".html" = "text/html; charset=utf-8"
    ".js"   = "text/javascript; charset=utf-8"
    ".mjs"  = "text/javascript; charset=utf-8"
    ".css"  = "text/css; charset=utf-8"
    ".json" = "application/json; charset=utf-8"
    ".wasm" = "application/wasm"
    ".whl"  = "application/octet-stream"
    ".zip"  = "application/zip"
    ".woff" = "font/woff"
    ".woff2" = "font/woff2"
    ".ttf"  = "font/ttf"
    ".svg"  = "image/svg+xml"
    ".png"  = "image/png"
    ".ico"  = "image/x-icon"
    ".txt"  = "text/plain; charset=utf-8"
    ".xlsx" = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
}

# First free port from 8765 upward, so a second copy does not collide.
$listener = $null
$port = $null
foreach ($p in 8765..8785) {
    try {
        $l = New-Object System.Net.HttpListener
        $l.Prefixes.Add("http://127.0.0.1:$p/")
        $l.Start()
        $listener = $l
        $port = $p
        break
    } catch { }
}
if (-not $listener) {
    Write-Host "  ERROR: no free port between 8765 and 8785." -ForegroundColor Red
    Read-Host "  Press Enter to close"
    exit 1
}

$url = "http://127.0.0.1:$port/"
Write-Host ""
Write-Host "  OnTarget - Diet Optimizer" -ForegroundColor Cyan
Write-Host "  ========================="
Write-Host ""
Write-Host "  Serving at $url"
Write-Host ""
Write-Host "  The browser should open by itself. If it does not, copy the"
Write-Host "  address above into Chrome, Edge or Firefox."
Write-Host ""
Write-Host "  The FIRST load takes 30-90 seconds while the page starts Python." -ForegroundColor Yellow
Write-Host "  A blank page during that time is normal."
Write-Host ""
Write-Host "  KEEP THIS WINDOW OPEN while you use the calculator." -ForegroundColor Yellow
Write-Host "  Close it (or press Ctrl+C) to stop."
Write-Host ""

Start-Process $url | Out-Null

try {
    while ($listener.IsListening) {
        $ctx = $listener.GetContext()
        $req = $ctx.Request
        $res = $ctx.Response
        try {
            # Strip the query string, decode %20 etc., map / to /index.html.
            $rel = [System.Uri]::UnescapeDataString($req.Url.AbsolutePath).TrimStart('/')
            if ($rel -eq "") { $rel = "index.html" }
            $full = Join-Path $siteDir ($rel -replace '/', '\')

            # Never serve anything outside site/ (defensive; the app never asks to).
            $resolved = [System.IO.Path]::GetFullPath($full)
            $siteFull = [System.IO.Path]::GetFullPath($siteDir)
            if (-not $resolved.StartsWith($siteFull, [System.StringComparison]::OrdinalIgnoreCase)) {
                $res.StatusCode = 403; $res.Close(); continue
            }
            if ((Test-Path $resolved) -and (Get-Item $resolved).PSIsContainer) {
                $resolved = Join-Path $resolved "index.html"
            }

            if (Test-Path $resolved) {
                $bytes = [System.IO.File]::ReadAllBytes($resolved)
                $ext = [System.IO.Path]::GetExtension($resolved).ToLower()
                $res.ContentType = if ($mime.ContainsKey($ext)) { $mime[$ext] } else { "application/octet-stream" }
                $res.ContentLength64 = $bytes.Length
                # The service worker must not be cached across versions.
                $res.Headers.Add("Cache-Control", "no-cache")
                $res.OutputStream.Write($bytes, 0, $bytes.Length)
                $res.StatusCode = 200
            } else {
                $res.StatusCode = 404
                $msg = [System.Text.Encoding]::UTF8.GetBytes("404 - not found: $rel")
                $res.OutputStream.Write($msg, 0, $msg.Length)
            }
        } catch {
            # A browser that cancels a download mid-flight lands here; keep serving.
            try { $res.StatusCode = 500 } catch { }
        } finally {
            try { $res.Close() } catch { }
        }
    }
} finally {
    $listener.Stop()
}
