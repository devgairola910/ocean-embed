"""Bundle index.html, index.css, and app.js into demo/standalone_view.html."""
import re
import os

def bundle():
    with open('index.html', 'r', encoding='utf-8') as f:
        html = f.read()
    with open('index.css', 'r', encoding='utf-8') as f:
        css = f.read()
    with open('app.js', 'r', encoding='utf-8') as f:
        js = f.read()

    # In index.html, replace CSS link with inline style
    html_bundled = re.sub(
        r'<link\s+rel=["\']stylesheet["\']\s+href=["\']index\.css["\']\s*/?>',
        f'<style>\n{css}\n</style>',
        html
    )
    # Replace JS script with inline script
    html_bundled = re.sub(
        r'<script\s+src=["\']app\.js["\']\s*></script>',
        f'<script>\n{js}\n</script>',
        html_bundled
    )

    os.makedirs('demo', exist_ok=True)
    with open('demo/standalone_view.html', 'w', encoding='utf-8') as f:
        f.write(html_bundled)

    print(f"Successfully generated demo/standalone_view.html ({len(html_bundled)} bytes)")

if __name__ == '__main__':
    bundle()
