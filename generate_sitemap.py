import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import quote

# ---------------- CONFIG ----------------
GITHUB_OWNER = "OmNagvekar"
GITHUB_REPO = "omnagvekar.github.io"
BLOGS_PATH = "blogs"  # folder containing your blog markdowns
BASE_URL = "https://omnagvekar.github.io/blog.html?post="
SITEMAP_FILE = "sitemap.xml"
# ----------------------------------------

def fetch_blog_files(owner, repo, path):
    """Fetch list of files from GitHub repo folder."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(url)
    response.raise_for_status()
    files = response.json()
    # Filter only markdown files
    md_files = [f for f in files if f['name'].endswith('.md')]
    return md_files

def generate_sitemap(files):
    """Generate sitemap.xml content."""
    urlset = ET.Element('urlset', xmlns="https://www.sitemaps.org/schemas/sitemap/0.9")

    # Add home page
    url = ET.SubElement(urlset, 'url')
    loc = ET.SubElement(url, 'loc')
    loc.text = "https://omnagvekar.github.io/"
    lastmod = ET.SubElement(url, 'lastmod')
    lastmod.text = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    changefreq = ET.SubElement(url, 'changefreq')
    changefreq.text = "weekly"
    priority = ET.SubElement(url, 'priority')
    priority.text = "1.0"

    # Add blogs
    for file in files:
        url = ET.SubElement(urlset, 'url')
        loc = ET.SubElement(url, 'loc')
        loc.text = BASE_URL + quote(file['name'])
        lastmod = ET.SubElement(url, 'lastmod')
        lastmod.text = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        changefreq = ET.SubElement(url, 'changefreq')
        changefreq.text = "monthly"
        priority = ET.SubElement(url, 'priority')
        priority.text = "0.8"

    tree = ET.ElementTree(urlset)
    tree.write(SITEMAP_FILE, encoding='utf-8', xml_declaration=True)
    print(f"Sitemap generated: {SITEMAP_FILE}")

if __name__ == "__main__":
    files = fetch_blog_files(GITHUB_OWNER, GITHUB_REPO, BLOGS_PATH)
    generate_sitemap(files)
