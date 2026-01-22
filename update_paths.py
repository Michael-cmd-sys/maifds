import os

file_path = r'\\wsl.localhost\Ubuntu\home\melch\projects\maifds_repo\ui\index.html'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update asset paths
    # Note: We need to be careful not to double replace if I ran it partially before, 
    # but since I am reading fresh, it should be fine.
    # Also need to handle "assets/" vs "assets/images" etc if they appear. 
    # But simple replace should cover most.
    
    content = content.replace('href="assets/', 'href="/landing_assets/')
    content = content.replace("href='assets/", "href='/landing_assets/")
    content = content.replace('src="assets/', 'src="/landing_assets/')
    content = content.replace("src='assets/", "src='/landing_assets/")
    
    # Update routing
    content = content.replace("href='app'", "href='/dashboard.html'")
    content = content.replace('href="app"', 'href="/dashboard.html"')
    
    # Also replace ./assets/ if it exists
    content = content.replace('href="./assets/', 'href="/landing_assets/')
    content = content.replace('src="./assets/', 'src="/landing_assets/')

    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("Successfully updated index.html")

except Exception as e:
    print(f"Error: {e}")
