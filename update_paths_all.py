import os
import glob

# Search for all HTML files in ui/
files = glob.glob(r'\\wsl.localhost\Ubuntu\home\melch\projects\maifds_repo\ui\*.html')

for file_path in files:
    if 'dashboard.html' in file_path:
        continue # Skip the react app entry point
        
    print(f"Processing {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update asset paths
        content = content.replace('href="assets/', 'href="/landing_assets/')
        content = content.replace("href='assets/", "href='/landing_assets/")
        content = content.replace('src="assets/', 'src="/landing_assets/')
        content = content.replace("src='assets/", "src='/landing_assets/")
        
        # Update routing
        content = content.replace("href='app'", "href='/dashboard.html'")
        content = content.replace('href="app"', 'href="/dashboard.html"')
        
        # Also replace ./assets/
        content = content.replace('href="./assets/', 'href="/landing_assets/')
        content = content.replace('src="./assets/', 'src="/landing_assets/')
        
        # Fix relative links to other html files if they are just "about.html" etc.
        # They should already work since they are all in root of ui/ (which is root of server).
        # But if they had relative links like "index.html", they are fine.
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"Successfully updated {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
