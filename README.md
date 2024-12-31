# wyag - A simple Git Implementation

`wyag` is a lightweight implementation of Git commands in Python. This project aims to provide a simple and educational implementation of core Git functionalities.

## Features

- Create and manage repositories
- Add and commit changes
- Create and manage branches
- Tagging
- Resolving references

## Usage

### Setup
On macOS/Linux you can add the path to `wyag` as an alias in ~/.bashrc
```
alias wyag="/path/to/wyag"
```
On windows, add the path in environment variables

All the commands are same that of `git`, just replace the `git` keyword with `wyag`

### Initialize a Repository
To initialize a new repository, use the following command:
```
wyag init
```

### Add Files
To add files to the staging area, use the add command:
```
wyag add <file1> <file2> ...
```

### Commit Changes
To commit changes to the repository, use the commit command:
```
wyag commit -m "commit message"
```


## Reference
Artucle: https://wyag.thb.lt/
GitHub: https://github.com/thblt/write-yourself-a-git

  
