import subprocess 






def git_pull():
    
    a  = subprocess.run(['git', 'pull'],
                    check = True,
                    capture_output = True,
                    text = True
                    )    


def commit_and_push(file_path:str, commit_message:str):

    
    subprocess.run(['git', 'commit', '-m', f'"{commit_message}"', file_path],
                    check = True,
                    capture_output = True,
                    text = True
                    )   

    subprocess.run(['git', 'push'],
                    check = True, 
                    capture_output = True,
                    text = True
                    )
    


def sync_photo(photo_path):
    
    git_pull()
    
    commit_and_push(file_path = photo_path, 
                    commit_message = 'New photo uploaded from API!'
                    )
    
    
    
    
    
                            
    

    