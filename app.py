"""Run a local browser demo with the Python standard library."""

from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs
from functools import lru_cache

from model import FEATURES, evaluate, score


@lru_cache(maxsize=1)
def benchmark_markup():
    return "".join(
        f"<dt>{name}</dt><dd>{value:.3f}</dd>" if isinstance(value, float)
        else f"<dt>{name}</dt><dd>{value}</dd>"
        for name, value in evaluate().items()
    )


def page(values, message="Enter a reading or try a preset."):
    inputs = "".join(
        f'<label for="reading-{i}">{name}</label>'
        f'<input id="reading-{i}" name="reading" type="number" '
        f'min="{low}" max="{high}" step="any" required value="{escape(str(value))}">'
        for i, ((name, _, low, high), value) in enumerate(zip(FEATURES, values))
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SensorWatch ML</title><style>
* {{ box-sizing: border-box }}
body {{ margin: 0; color: #e4ebed; background: #101c21; font: 1rem/1.6 system-ui; }}
main {{ max-width: 54rem; padding: clamp(1rem,5vw,3rem); margin: auto; }}
h1 {{ font-size: clamp(2.3rem,6vw,4rem); margin: 0; line-height: 1.1; }}
h2 {{ margin-top: 0; }}
section {{ padding: 1.25rem; border: 1px solid #395057; border-radius: .75rem;
background: #17282e; margin: 1.4rem 0; }}
form, dl {{ display: grid; grid-template-columns: 1fr 1fr; gap: .65rem; }}
label {{ align-self: center; }}
dd {{ margin: 0; font-variant-numeric: tabular-nums; }}
input, button, a {{ font: inherit; }}
input {{ width:100%; color: inherit; background: #101c21; border: 1px solid #698087;
padding: .45rem; border-radius: .35rem; }}
button {{ grid-column: 1 / -1; background: #a7f5d2; color: #10261f;
padding: .5rem; border: 0; border-radius: .35rem; font-weight: 650; cursor:pointer; }}
a {{ color: #a7f5d2; margin-right: .65rem; }}
output {{ display: block; margin-top: 1rem; font-weight: 650; }}
small {{ color: #b5c9cf; }}
</style></head><body><main>
<h1>SensorWatch ML</h1>
<p>Flag unusual machine readings with an Isolation Forest trained on synthetic telemetry.</p>
<section><h2>Score a reading</h2>
<p><a href="/score?reading=62&amp;reading=2&amp;reading=8">Typical reading</a>
<a href="/score?reading=77&amp;reading=3.4&amp;reading=11">Fault example</a></p>
<form action="/score" method="get">{inputs}<button>Score reading</button></form>
<output role="status">{escape(message)}</output>
<small>Positive scores flag review. The score is a model margin, not a probability.</small>
</section>
<section><h2>Synthetic benchmark</h2>
<p>200 normal readings and 60 simulated faults held out from training.</p>
<dl>{benchmark_markup()}</dl>
</section>
</main></body></html>""".encode("utf-8")


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path not in {"/", "/score"}:
            self.send_error(404)
            return
        values = [default for _, default, _, _ in FEATURES]
        message = "Enter a reading or try a preset."
        status = 200
        if path == "/score":
            try:
                readings = parse_qs(query, max_num_fields=3).get("reading", [])
                margin, flagged = score(readings)
                values = readings
                label = "Review suggested" if flagged else "Within baseline"
                message = f"{label} · score {margin:+.4f}"
            except ValueError:
                status = 400
                message = "Enter three numeric readings within the displayed limits."
        body = page(values, message)
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    with ThreadingHTTPServer(("127.0.0.1", 8000), DemoHandler) as server:
        print("Open http://127.0.0.1:8000", flush=True)
        server.serve_forever()
