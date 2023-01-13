#Starting the project
Preparing the structure and organizing the project.

## Installation
### With Conda
Creating a conda environment:
```
conda create --name aidl-final-project python=3.8
```
To activate the environment:
```
conda activate aidl-final-project
```
and install the dependencies
```
pip install -r requirements.txt
```

To stop/deactivate the environment:
```
conda deactivate
```


## GIT
### Configure Git
Repo: https://github.com/vmaiol/aidl-final-project

Checking Git is installed
```
git --version
```

To start using Git from your computer, you must enter your credentials to identify yourself
```
git config --global user.name "your_username"
```
Add your email address
```
git config --global user.email "your_email_address@example.com"
```

To check the configuration
```
git config --global --list
```

### Generating SSH Keys to connect to the private repo (in MAC)
In terminal:
```
ssh-keygen
```
And then:
- Go to the project homepage or section in Github.
- Code -> SSH -> Add a new public key
- New Key
- We copy paste the public key that is inside the file we have just created ended with .pub and that was created in our laptop when "ssh-keygen".
- !!!!! By default, I think the SSH connection to Github from our laptop, uses a file called id_rsa (if you use Mac, it is inside /Users/--user-name--/.ssh/). In my case, as I already have another SHH key in that file and I didn't want to override it, I saved the keys in another filename (in the ssh-keygen process it asks for it). Then, you need to create a configuration file where you specify which file to read every time you try to connect to X host via SSH, in this case Github. Like this:
```
Host github.com
 HostName github.com
 IdentityFile ~/.ssh/aidl_final_project_github
 ```
 - The connection can be tested like this:
```
 ssh -vT git@github.com
```

### Cloning the repository (ssh)
```
git clone git@gitlab.com:gitlab-tests/sample-project.git
```

### Convert a local directory into a repository
Initialize a local folder so Git tracks it as a repository
```
git init
```

Adding a “remote” to tell Git which remote repository in GitLab is tied to the specific local folder.
In the directory you’ve initialized
```
git remote add origin git@gitlab.com:username/projectpath.git
```

To view remote repositories:
```
git remote -v
```

### Download the latest changes in the project
To work on an up-to-date copy of the project, you pull to get all the changes made by users since the last time you cloned or pulled the project. Replace <name-of-branch> with the name of your default branch to get the main branch code, or replace it with the branch name of the branch you are currently working in
```
git pull <REMOTE> <name-of-branch>
```
REMOTE is typically origin. This is where the repository was cloned from, and it indicates the SSH or HTTPS URL of the repository on the remote server. <name-of-branch> is usually the name of your default branch, but it may be any existing branch


### Add and commit local changes
To stage a file for commit
```
git add <file-name OR folder-name>
```

Or, to stage all files in the current directory and subdirectory, type
```
git add .
```

Confirm that the files have been added to staging. The files should be displayed in green text.
```
git status
```

To commit the staged files
```
git commit -m "COMMENT TO DESCRIBE THE INTENTION OF THE COMMIT"
```

Stage and commit all changes
As a shortcut, you can add all local changes to staging and commit them with one command:
```
git commit -a -m "COMMENT TO DESCRIBE THE INTENTION OF THE COMMIT"
```

### Send changes
To push all local changes to the remote repository:
```
git push <remote> <name-of-branch>
```

For example, to push your local commits to the main branch of the origin remote:
```
git push origin main
```

### Ignoring files/folders for the commit
As it is is quite "personal" and has many combinations/possibilities, I leave here a basic "quick read" guide to implement .gitignore:
https://www.freecodecamp.org/news/gitignore-file-how-to-ignore-files-and-folders-in-git/


For more "general" detailed info -> https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html
