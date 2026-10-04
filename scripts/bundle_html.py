"""Bulletproof standalone HTML bundler for OceanEmbed dashboard.

Inlines all CSS, JS, and converts local assets (images/fonts) into base64 Data URIs
so demo/standalone_view.html works 100% self-contained inside Streamlit Cloud iframe.
"""
import os
import base64
import re

def get_base64_data_uri(file_path: str) -> str:
    """Return base64 Data URI for a given file."""
    if not os.path.exists(file_path):
        return file_path
    ext = os.path.splitext(file_path)[1].lower()
    mime = "image/png"
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".svg":
        mime = "image/svg+xml"
    elif ext == ".webp":
        mime = "image/webp"
        
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"

def bundle():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    with open(os.path.join(project_root, "index.html"), "r", encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(project_root, "index.css"), "r", encoding="utf-8") as f:
        css = f.read()
    with open(os.path.join(project_root, "app.js"), "r", encoding="utf-8") as f:
        js = f.read()

    # Asset paths to inline as Base64 Data URIs
    asset_files = [
        "assets/underwater_bg.jpg",
        "assets/clownfish.png",
        "assets/yellow_fish.png",
        "assets/blue_fish.png",
        "assets/sample_sst_satellite.jpg",
        "assets/sample_cyclone_satellite.jpg"
    ]

    for asset in asset_files:
        full_path = os.path.join(project_root, asset)
        if os.path.exists(full_path):
            data_uri = get_base64_data_uri(full_path)
            html = html.replace(asset, data_uri)
            css = css.replace(asset, data_uri)
            js = js.replace(asset, data_uri)
            print(f"✅ Inlined {asset} ({len(data_uri)} chars)")

    # Safe CSS inlining (find any <link ... index.css... > tag)
    css_pattern = re.compile(r'<link\s+[^>]*href=["\'][^"\']*index\.css[^"\']*["\'][^>]*>', re.IGNORECASE)
    match = css_pattern.search(html)
    if match:
        html = html[:match.start()] + f"<style>\n{css}\n</style>" + html[match.end():]
        print("✅ Inlined index.css into <style> block")
    else:
        # Fallback: insert before </head>
        html = html.replace("</head>", f"<style>\n{css}\n</style>\n</head>")
        print("⚠️ Fallback: Injected index.css before </head>")

    # Safe JS inlining (find <script ... src="app.js"...>)
    js_pattern = re.compile(r'<script\s+[^>]*src=["\'][^"\']*app\.js[^"\']*["\'][^>]*>\s*</script>', re.IGNORECASE)
    match_js = js_pattern.search(html)
    if match_js:
        html = html[:match_js.start()] + f"<script>\n{js}\n</script>" + html[match_js.end():]
        print("✅ Inlined app.js into <script> block")
    else:
        html = html.replace("</body>", f"<script>\n{js}\n</script>\n</body>")
        print("⚠️ Fallback: Injected app.js before </body>")

    output_path = os.path.join(project_root, "demo", "standalone_view.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n🎉 Successfully created standalone self-contained bundle: {output_path} ({len(html):,} bytes)")

if __name__ == "__main__":
    bundle()
