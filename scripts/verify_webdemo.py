"""Phase 4 verification: load the demo, confirm ONNX model loads and the agent plays."""
import sys
import time

from playwright.sync_api import sync_playwright

URL = "http://localhost:8321"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    errors, logs = [], []
    page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(URL)
    page.wait_for_load_state("networkidle")
    time.sleep(4)  # let the model load and the game run a few dozen ticks

    state = page.evaluate("""() => {
        const g = (id) => { const el = document.getElementById(id);
                            return el ? el.textContent : null; };
        const banners = [...document.querySelectorAll('.banner, [class*=banner], [id*=banner]')]
            .filter(b => b.offsetParent !== null).map(b => b.textContent.trim());
        return {
            title: document.title,
            visibleBanners: banners,
            bodySnippet: document.body.innerText.slice(0, 400),
            hasCanvas: !!document.querySelector('canvas'),
        };
    }""")
    page.screenshot(path="results/webdemo_check.png")

    # sample the game twice to prove the simulation advances
    s1 = page.evaluate("() => document.body.innerText")
    time.sleep(2)
    s2 = page.evaluate("() => document.body.innerText")

    print("TITLE:", state["title"])
    print("CANVAS:", state["hasCanvas"])
    print("VISIBLE BANNERS:", state["visibleBanners"])
    print("BODY:", " | ".join(state["bodySnippet"].split("\n")[:8]))
    print("PAGE ERRORS:", errors if errors else "none")
    bad = [l for l in logs if "error" in l.lower() and "favicon" not in l.lower()]
    print("CONSOLE ERRORS:", bad if bad else "none")
    print("SIM ADVANCES:", s1 != s2)
    browser.close()

    ok = (state["hasCanvas"] and not errors and not bad
          and not any("not loaded" in b or "missing" in b
                      for b in state["visibleBanners"]))
    print("WEBDEMO_OK" if ok else "WEBDEMO_ISSUES")
    sys.exit(0 if ok else 1)
