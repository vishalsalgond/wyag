import argparse
import collections
import configparser
from datetime import datetime
import grp, pwd
import hashlib
from math import ceil
import os
import re
import sys
import zlib

argparser = argparse.ArgumentParser()
argsubparsers = argparser.add_subparsers(title="Commands", dest="command")
# argsubparsers.required = True

# wyag init
argsp = argsubparsers.add_parser("init", help="Initialize a new, empty git repository.")
argsp.add_argument("path", metavar="directory", nargs="?", default=".", help="Where to create the git repository.")

# wyag cat-file TYPE OBJECT
argsp = argsubparsers.add_parser("cat-file", help="Provide content of repository objects.")
argsp.add_argument("type", metavar="type", choices=["blob", "commit", "tag", "tree"], help="Specify the object type.")
argsp.add_argument("object", metavar="object", help="The object to display.")

# wyag hash-object [-w] [-t TYPE] FILE
argsp = argsubparsers.add_parser("hash-object", help="Compute object id and optionally create a blob from a file")
argsp.add_argument("-t",
    metavar="type",
    dest="type",
    choices=["blob", "commit", "tag", "tree"],
    default="blob",
    help="Specify the type of the object.")
argsp.add_argument("-w",
    dest="write",
    action="store_true",
    help="Write the object into the repository.")
argsp.add_argument("path", help="Read object from <file>")


# wyag log
argsp = argsubparsers.add_parser("log", help="Display history of given commit.")
argsp.add_argument("commit",
    default="HEAD",
    nargs="?",
    help="Commit to start at.")

# wyag ls-tree [-r] TREE
argsp = argsubparsers.add_parser("ls-tree", help="Pretty-print a tree object")
argsp.add_argument("-r", dest="recursive", action="store_true", help="Recurse into sub-trees")
argsp.add_argument("tree", help="A tree object")

# wyag checkout
argsp = argsubparsers.add_parser("checkout", help="Checkout a commit inside of a directory.")
argsp.add_argument("commit", help="The commit or tree to checkout.")
argsp.add_argument("path", help="The empty directory to checkout on.")

# wyag show-ref
argsp = argsubparsers.add_parser("show-ref", help="List references.")

# wyag tag
argsp = argsubparsers.add_parser("tag", help="List and create tags.")
argsp.add_argument("-a", action="store_true", dest="create_tag_object",
    help="Whether to create a tag object.")
argsp.add_argument("name", nargs="?", help="The new tag name.")
argsp.add_argument("object", default="HEAD", nargs="?",
    help="The object to which the new tag will point to.")

argsp = argsubparsers.add_parser("rev-parse", help="Parse revision (or other versions) or identifiers")
argsp.add_argument("--wyag-type",
    metavar="type",
    dest="type",
    choices=["blob", "commit", "tree", "tag"],
    default=None,
    help="Specify the expected type")
argsp.add_argument("name", help="The name to parse")


argsp = argsubparsers.add_parser("ls-files", help="List all the files in staging area")
argsp.add_argument("--verbose", action="store_true", help="Show everything.")

argsp = argsubparsers.add_parser("check-ignore", help="Check paths against ingore rules.")
argsp.add_argument("path", nargs="+", help="Paths to check")

argsp = argsubparsers.add_parser("status", help="Show the working tree status.")

argsp = argsubparsers.add_parser("rm", help="Remove files from the working tree and the index.")
argsp.add_argument("path", nargs="+", help="Files to remove")

argsp = argsubparsers.add_parser("add", help = "Add files to the staging area.")
argsp.add_argument("path", nargs="+", help="Files to add")

argsp = argsubparsers.add_parser("commit", help="Record changes to the repository.")
argsp.add_argument("-m", metavar="message", dest="message",
    help="Message to associate with this commit.")

# Helper functions

def main(argv=sys.argv[1:]):
    args = argparser.parse_args(argv)
    match args.command:
        case "add"          : cmd_add(args)
        case "cat-file"     : cmd_cat_file(args)
        case "check-ignore" : cmd_check_ignore(args)
        case "checkout"     : cmd_checkout(args)
        case "commit"       : cmd_commit(args)
        case "hash-object"  : cmd_hash_object(args)
        case "init"         : cmd_init(args)
        case "log"          : cmd_log(args)
        case "ls-files"     : cmd_ls_files(args)
        case "ls-tree"      : cmd_ls_tree(args)
        case "rev-parse"    : cmd_rev_parse(args)
        case "rm"           : cmd_rm(args)
        case "show-ref"     : cmd_show_ref(args)
        case "status"       : cmd_status(args)
        case "tag"          : cmd_tag(args)
        case _              : print("Inavlid command!")


def repo_path(repo, *path):
    """Compute path inside repo's .git directory"""
    return os.path.join(repo.gitdir, *path)

def repo_file(repo, *path, make_directory=False):
    if repo_dir(repo, *path[:-1], make_directory=make_directory):
        return repo_path(repo, *path)

def repo_dir(repo, *path, make_directory=False):
    path = repo_path(repo, *path)

    if os.path.exists(path):
        if os.path.isdir(path):
            return path
        else:
            raise Exception("[ERROR]: {} is not a directory.".format(path))

    if make_directory:
        os.makedirs(path)
        return path
    
    return None


def repo_create(path):
    """Create a new git repository at the given path"""

    repo = GitRepository(path, force=True)

    if os.path.exists(repo.worktree):
        if not os.path.isdir(repo.worktree):
            raise Exception("[ERROR]: {} is not a directory.".format(repo.worktree))
        if os.path.exists(repo.gitdir) and os.listdir(repo.gitdir):
            raise Exception("[ERROR: {} is already a git repository.".format(repo.worktree))
    else:
        # TODO: check if we can use repo_dir here
        os.makedirs(repo.worktree)  

    # create the necessary directories
    assert repo_dir(repo, "branches", make_directory=True)
    assert repo_dir(repo, "objects", make_directory=True)
    assert repo_dir(repo, "refs", "tags", make_directory=True)
    assert repo_dir(repo, "refs", "heads", make_directory=True)

    # .git/description
    with open(repo_file(repo, "description"), "w") as f:
        f.write("Unnamed repository: edit this file 'description' to name the repository.\n")

    # .git/HEAD
    with open(repo_file(repo, "HEAD"), "w") as f:
        f.write("ref: refs/heads/master\n")

    with open(repo_file(repo, "config"), "w") as f:
        config = repo_default_config()
        config.write(f)


def repo_default_config():
    ret = configparser.ConfigParser()

    ret.add_section("core")
    ret.set("core", "repositoryformatversion", "0")
    ret.set("core", "filemode", "false")
    ret.set("core", "bare", "false")

    return ret

def repo_find(path=".", required=True):
    path = os.path.realpath(path)

    if os.path.isdir(os.path.join(path, ".git")):
        return GitRepository(path)

    # Find the repo in parent directory recursively
    parent = os.path.realpath(os.path.join(path, ".."))

    # Base case for recursion
    if parent == path:
        if required:
            raise Exception("[ERROR]: No git directory found.")
        else:
            return None

    return repo_find(parent, required)

def object_read(repo, sha):
    """Read the hashed object from the git repo. Return a 
       GitObject whose exact type depends on the object."""
    
    path = repo_file(repo, "objects", sha[0:2], sha[2:])

    if not os.path.isfile(path):
        return None
    
    with open(path, "rb") as f:
        raw = zlib.decompress(f.read())

    # Read object type
    type_end_index = raw.find(b' ')
    fmt = raw[:type_end_index]

    # Read and validate object size
    size_end_index = raw.find(b'\x00', type_end_index)
    size = int(raw[type_end_index : size_end_index].decode("ascii"))
    if size != len(raw) - size_end_index - 1:
        raise Exception("Malformed object {0}: Bad length".format(sha))

    # print(fmt)
    # Find the constructor:
    match fmt:
        case b'commit' : constructor = GitCommit
        case b'tree' : constructor = GitTree
        case b'blob' : constructor = GitBlob
        case b'tag' : constructor = GitTag
        case _:
            raise Exception("Unknown object type {0} for object {1}".format(fmt.decode("ascii"), sha))
        
    # Call the constructor and return the object
    return constructor(raw[size_end_index + 1:])


def object_write(obj, repo=None):
    # Serialize object data
    data = obj.serialize()
    # print(obj.items[0].sha, data)
    # Add header
    result = obj.fmt + b' ' + str(len(data)).encode() + b'\x00' + data

    # Compute hash (part 1 : dir + part 2 : name of the file)
    sha = hashlib.sha1(result).hexdigest()

    if repo:
        path = repo_file(repo, "objects", sha[0:2], sha[2:], make_directory=True)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                # Compress and write to the file
                f.write(zlib.compress(result))
    
    return sha

def object_find(repo, name, fmt=None, follow=True):
    sha = object_resolve(repo, name)

    if not sha:
        raise Exception("No such reference {0}.".format(name))
    
    if len(sha) > 1:
        raise Exception("Ambiguous reference {0}: Candidates are:\n - {1}".format(name, "\n - ".join(sha)))

    sha = sha[0]
    if not fmt:
        return sha

    while True:
        # print("Resolved sha: ", sha)
        obj = object_read(repo, sha)

        if obj.fmt == fmt:
            return sha
        
        if not follow:
            return None
        
        # Follow tags
        if obj.fmt == b'tag':
            sha = obj.kvlm[b'object'].decode("ascii")
        elif obj.fmt == b'commit' and fmt == b'tree':
            sha = obj.kvlm[b'tree'].decode("ascii")
        else:
            return None


def cat_file(repo, obj, fmt=None):
    obj = object_read(repo, object_find(repo, obj, fmt=fmt))
    sys.stdout.buffer.write(obj.serialize())

def object_hash(fd, fmt, repo=None):
    data = fd.read()

    match fmt:
        case b'commit' : obj = GitCommit(data)
        case b'blob'   : obj = GitBlob(data)
        case b'tree'   : obj = GitTree(data)
        case b'tag'    : obj = GitTag(data)
        case _: raise Exception("Unknown type {}".format(fmt))

    return object_write(obj, repo) 

def kvlm_parse(raw, start=0, dct=None):
    if not dct:
        dct = collections.OrderedDict()

    # Search for next space and next newline
    spc = raw.find(b' ', start)
    nl = raw.find(b'\n', start)

    # Base case - read the commit message
    if (spc < 0) or (nl < spc):
        assert nl == start
        dct[None] = raw[start + 1:]
        return dct

    # Get the key
    key = raw[start:spc]

    # Get the value
    end = start
    while True:
        end = raw.find(b'\n', end + 1)
        if raw[end + 1] != ord(' '):
            break
    
    value = raw[spc + 1: end].replace(b'\n ', b'\n')

    if key in dct:
        if type(dct[key]) == list:
            dct[key].append(value)
        else:
            dct[key] = [dct[key], value]
    else:
        dct[key] = value

    return kvlm_parse(raw, start=end + 1, dct=dct)


def kvlm_serialize(kvlm):
    ret = b''

    # Key-Value fields
    for k in kvlm.keys():
        if k == None:
            continue

        val = kvlm[k]
        if type(val) != list:
            val = [val]

        for v in val:
            ret += k + b' ' + (v.replace(b'\n', b'\n ')) + b'\n'

    # Append message
    ret += b'\n' + kvlm[None] + b'\n'

    return ret


def log_graphviz(repo, sha, seen):
    if sha in seen:
        return
    seen.add(sha)

    commit = object_read(repo, sha)
    short_hash = sha[0:8]
    message = commit.kvlm[None].decode("utf8").strip()
    message = message.replace("\\", "\\\\")
    message = message.replace("\"", "\\\"")

    # Print only the first line
    if "\n" in message:
        message = message[:message.index("\n")]

    print(" commit {0} [label=\"{1}: {2}\"]".format(sha, sha[0:7], message))
    print(" Author: {}".format(commit.kvlm[b'author']))
    print("     {}".format(message))
    assert commit.fmt == b'commit'

    # Base case: Initial commit
    if not b'parent' in commit.kvlm.keys():
        return

    parents = commit.kvlm[b'parent']

    if type(parents) != list:
        parents = [parents]
    
    print()

    for p in parents:
        p = p.decode("ascii")
        print(" c_{0} -> c{1}".format(sha, p))
        log_graphviz(repo, p, seen)


def tree_parse_one(raw, start=0):
    x = raw.find(b' ', start)
    assert x - start == 5 or x - start == 6

    mode = raw[start : x]
    if len(mode) == 5:
        # Normalize to six bytes
        mode = b" " + mode
    
    # Find the null terminator after the path
    y = raw.find(b'\x00', x)
    # Read the path
    path = raw[x + 1 : y]

    # Read the SHA and convert to hex string
    sha = format(int.from_bytes(raw[y + 1 : y + 21], "big"), "040x")
    return y + 21, GitTreeLeaf(mode, path.decode("utf8"), sha)


def tree_parse(raw):
    pos = 0
    max_pos = len(raw)
    res = list()

    while pos < max_pos:
        pos, leaf = tree_parse_one(raw, pos)
        res.append(leaf)
    
    return res
    
def tree_leaf_sort_key(leaf):
    if leaf.mode.startswith(b"10"):
        return leaf.path
    else:
        return leaf.path + "/"

def tree_serialize(obj):
    obj.items.sort(key=tree_leaf_sort_key)
    res = b''

    for leaf in obj.items:
        # Write mode
        res += leaf.mode
        res += b' '

        # Write path
        res += leaf.path.encode("utf8")
        res += b'\x00'

        # Write SHA
        sha = int(leaf.sha, 16)
        res += sha.to_bytes(20, byteorder="big")
    
    return res


def ls_tree(repo, ref, recursive=None, prefix=""):
    sha = object_find(repo, ref, fmt=b"tree")
    obj = object_read(repo, sha)

    for item in obj.items:
        if len(item.mode) == 5:
            type = item.mode[:1]
        else:
            type = item.mode[:2]
        
        match type:
            case b'04': type = "tree" # TODO: check if this should be leaf
            case b'10': type = "blob" # A regular file
            case b'12': type = "blob" # A symlink (TODO)
            case b'16': type = "commit"
            case _: raise Exception("Unexpected tree leaf node {}".format(item.mode))

        if not(recursive and type == "tree"): # This is a leaf, print
            print("{0} {1}\t{2}".format(
                "0" * (6 - len(item.mode)) + item.mode.decode("ascii"),
                type,
                item.sha,
                os.path.join(prefix, item.path)
            ))
        else: # This is a branch, recurse
            ls_tree(repo, item.sha, recursive, os.path.join(prefix, item.path))


def tree_checkout(repo, tree, path):
    for item in tree.items:
        obj = object_read(repo, item.sha)
        dest = os.path.join(path, item.path)

        if obj.fmt == b'tree':
            os.mkdir(dest)
            tree_checkout(repo, obj, dest)
        elif obj.fmt == b'blob':
            with open(dest, 'wb') as f:
                f.write(obj.blobdata)
        
def ref_resolve(repo, ref):
    # print("Ref resolve for :", ref)
    path = repo_file(repo, ref)
    # print("Path: ", path)

    if not os.path.isfile(path):
        return None
    
    with open(path, 'r') as fp:
        data = fp.read()[:-1]  # Drop /n
    
    # print("Data : ", data)
    if data.startswith("ref: "):
        return ref_resolve(repo, data[5:])
    else:
        return data

def ref_list(repo, path=None):
    if not path:
        path = repo_dir(repo, "refs")
    
    ret = collections.OrderedDict()
    for f in sorted(os.listdir(path)):
        full_path = os.path.join(path, f)
        if os.path.isdir(full_path):
            ret[f] = ref_list(repo, full_path)
        else:
            ret[f] = ref_resolve(repo, full_path)
    
    return ret


def show_ref(repo, refs, with_hash=True, prefix=""):
    for k, v in refs.items():
        if type(v) == str:
            print("{0}{1}{2}".format(
                v + " " if with_hash else "",  # TODO: Understand
                prefix + "/" if prefix else "",
                k
            ))
        else:
            show_ref(repo, v, with_hash=with_hash, prefix="{0}{1}{2}".format(
                prefix, "/" if prefix else "", k))


def tag_create(repo, name, ref, create_tag_object=False):
    sha = object_find(repo, ref)
    
    if create_tag_object:
        # create tag object (commit)
        tag = GitTag(repo)
        tag.kvlm = collections.OrderedDict()
        tag.kvlm[b'object'] = sha.encode()
        tag.kvlm[b'type'] = b'commit'
        tag.kvlm[b'tag'] = name.encode()
        tag.kvlm[b'tagger'] = b'user <user@example.com>'
        tag.kvlm[None] = b"A tag generated by user, you cannot currently add custom message!" # TODO
        tag_sha = object_write(tag)

        ref_create(repo, "tags/" + name, tag_sha)
    else:
        # create lightweight tag (ref)
        ref_create(repo, "tags/" + name, sha)


def ref_create(repo, ref_name, sha):
    with open(repo_file(repo, "refs/" + ref_name), 'w') as fp:
        fp.write(sha + "\n")

def object_resolve(repo, name):
    """Resolve name to an object hash in the repo.
    The function can resolve:
    - the HEAD literal
    - short and long hashes
    - tags
    - branches
    - remote branches
    """
    candidates = list()
    hashRE = re.compile(r"^[0-9A-Fa-f]{4,40}$")

    # Return if empty string
    if not name.strip():
        return None

    if name == "HEAD":
        return [ref_resolve(repo, "HEAD")]
    
    # If it is a hex string, try for a hash
    if hashRE.match(name):
        # This may be a hash either full or short
        # 4 is the minimum length for a short hash
        name = name.lower()
        prefix = name[0:2]
        path = repo_dir(repo, "objects", prefix, make_directory=False)
        if path:
            rem = name[2:]
            for f in os.listdir(path):
                if f.startswith(rem):
                    candidates.append(prefix + f)
    
    # Try for references
    as_tag = ref_resolve(repo, "refs/tags/" + name)
    if as_tag: 
        candidates.append(as_tag)
    
    as_branch = ref_resolve(repo, "refs/heads/" + name)
    if as_branch:
        candidates.append(as_branch)

    return candidates


def index_read(repo):
    index_file = repo_file(repo, "index")
    endianness = "big"

    # New repos have no index
    if not os.path.exists(index_file):
        return GitIndex()

    with open(index_file, 'rb') as f:
        raw = f.read()
    
    header = raw[:12] # first 12 bytes is header
    signature = header[:4]
    # assert signature == b'DIRC' # stands for "DirCache"
    version = int.from_bytes(header[4:8], endianness)
    # assert version == 2, "wyag only supports index file version 2"
    count = int.from_bytes(header[8:12], endianness)

    entries = list() # Entries for GitIndex
    content = raw[12:]
    idx = 0
    for i in range(0, count):
        # Read creation time, a unix timestamp (in secs and nanosecs)
        ctime_s = int.from_bytes(content[idx : idx+4], endianness)
        ctime_ns = int.from_bytes(content[idx+4 : idx+8], endianness)

        # Read modification time
        mtime_s = int.from_bytes(content[idx+8 : idx+12], endianness)
        mtime_ns = int.from_bytes(content[idx+12 : idx+16], endianness)

        dev = int.from_bytes(content[idx:16 : idx+20], endianness)
        ino = int.from_bytes(content[idx+20 : idx+24], endianness)

        unused = int.from_bytes(content[idx+24 : idx+24], endianness)
        assert 0 == unused
        
        mode = int.from_bytes(content[idx+26 : idx+28], endianness)
        mode_type = mode >> 12

        assert mode_type in [0b1000, 0b1010, 0b1110]
        mode_perms = mode & 0b0000000111111111

        uid = int.from_bytes(content[idx+28 : idx+32], endianness)
        gid = int.from_bytes(content[idx+32 : idx+36], endianness)
        fsize = int.from_bytes(content[idx+36 : idx+40], endianness)
        sha = format(int.from_bytes(content[idx+40 : idx+60], endianness), "040x")
        flags = int.from_bytes(content[idx+60 : idx+62], endianness)

        flag_assume_valid = (flags & 0b1000000000000000) != 0
        flag_extended = (flags & 0b0100000000000000) != 0
        assert not flag_extended
        flag_stage =  flags & 0b0011000000000000
        name_length = flags & 0b0000111111111111 # TODO

        # Update the iterator
        idx += 62

        if name_length < 0xFFF:
            assert content[idx + name_length] == 0x00
            raw_name = content[idx : idx + name_length]
            idx += name_length + 1
        else:
            print("Notice: Name is 0x{:X} bytes long".format(name_length))
            null_idx = content.find(b'\0x00', idx + 0xFFF)
            raw_name = content[idx : null_idx]
            idx = null_idx + 1

        name = raw_name.decode("utf8")

        # Data is padded on multiples of 8
        idx = 8 * ceil(idx / 8)

        entries.append(GitIndexEntry(ctime=(ctime_s, ctime_ns), mtime=(mtime_s, mtime_ns),
                                    dev=dev, ino=ino, mode_type=mode_type, mode_perms=mode_perms,
                                    uid=uid, gid=gid, fsize=fsize, sha=sha, flag_assume_valid=flag_assume_valid,
                                    flag_stage=flag_stage, name=name))

    return GitIndex(version=version, entries=entries)


def gitignore_parse1(raw):
    raw = raw.strip()

    if not raw or raw[0] == "#":
        return None
    elif raw[0] == "!":
        return (raw[1:], False)
    elif raw[0] == "\\":
        return (raw[1:], True)
    else:
        return (raw, True)


def gitignore_parse(lines):
    ret = list()

    for line in lines:
        parsed = gitignore_parse1(line)
        if parsed:
            ret.append(parsed)

    return ret

# TODO
def gitignore_read(repo):
    ret = GitIgnore(absolute=list(), scoped=list())

    # Read local configuration in .git/info/exclude
    repo_file = os.path.join(repo.gitdir, "info/exclude")
    if os.path.exists(repo_file):
        with open(repo_file, "r") as f:
            ret.absolute.append(gitignore_parse(f.readlines()))

    # Global configuration
    if "XDG_CONFIG_HOME" in os.environ:
        config_home = os.environ["XDG_CONFIG_HOME"]
    else:
        config_home = os.path.expanduser("~/.config")
    
    global_file = os.path.join(config_home, "git/ignore")
    if os.path.exists(global_file):
        with open(global_file, "r") as f:
            ret.absolute.append(gitignore_parse(f.readlines()))

    # .gitignore files in the index
    index = index_read(repo)

    for entry in index.entries:
        if entry.name == ".gitignore" or entry.name.endswith("./gitignore"):
            dir_name = os.path.dirname(entry.name)
            contents = object_read(entry.name)
            lines = contents.blobdata("utf8").splitlines()
            ret.scoped[dir_name] = gitignore_parse(lines)

    return ret


def check_ignore1(rules, path):
    result = None
    for (pattern, value) in rules:
        if fnmatch(path, pattern):
            result = value
    return result

def check_ignore_scoped(rules, path):
    parent = os.path.dirname(path)
    while True:
        if parent in rules:
            result = check_ignore1(rules[parent], path)
            if result != None:
                return result
        if parent == "":
            break
        parent = os.path.dirname(parent)
    return None

def check_ignore_absolute(rules, path):
    parent = os.path.dirname(path)
    for ruleset in rules:
        result = check_ignore1(ruleset, path)
        if result != None:
            return result
    return False


def check_ignore(rules, path):
    if os.path.isabs(path):
        raise Exception("This function requires path to be relative to the repository's root.")
    
    result = check_ignore_scoped(rules.scoped, path)
    if result != None:
        return result
    
    return check_ignore_absolute(rules.absolute, path)

def branch_get_active(repo):
    with open(repo_file(repo, "HEAD"), "r") as f:
        head = f.read()
    
    if head.startswith("ref: refs/heads/"):
        return (head[16 : -1])
    else:
        return False

def cmd_status_branch(repo):
    branch = branch_get_active(repo)
    if branch:
        print("On branch {}".format(branch))
    else:
        print("HEAD detached at {}".format(object_find(repo, "HEAD")))


def tree_to_dict(repo, ref, prefix=""):
    ret = dict()
    tree_sha = object_find(repo, ref, fmt=b"tree")
    tree = object_read(repo, tree_sha)

    for leaf in tree.items:
        full_path = os.path.join(prefix, leaf.path)

        is_subtree = leaf.mode.startswith(b'04')

        if is_subtree:
            ret.update(tree_to_dict(repo, leaf.sha, full_path))
        else:
            ret[full_path] = leaf.sha
    
    return ret

def cmd_status_head_index(repo, index):
    print("Changes to be committed:")

    head = tree_to_dict(repo, "HEAD")
    for entry in index.entries:
        if entry.name in head:
            if head[entry.name] != entry.sha:
                print("    modified:", entry.name)
            del head[entry.name]
        else:
            print("    added:    ", entry.name)
    
    # Keys still in HEAD and not in index have been deleted
    for entry in head.keys():
        print("    deleted:", entry.name)


def cmd_status_index_worktree(repo, index):
    print("Changes not staged for commit:")

    ignore = gitignore_read(repo)
    gitdir_prefix = repo.gitdir + os.path.sep
    all_files = list()

    # Walk the filesystem and find out all the files
    for (root, _, files) in os.walk(repo.worktree, True):
        if root == repo.gitdir or root.startswith(gitdir_prefix):   # TODO
            continue

        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, repo.worktree)
            all_files.append(rel_path)

    for entry in index.entries:
        full_path = os.path.join(repo.worktree, entry.name)

        if not os.path.exists(full_path):
            print("    deleted: ", entry.name)
        else:
            stat = os.stat(full_path)

            # Compare metadata to find out if the file has been modified
            ctime_ns = entry.ctime[0] * 10**9 + entry.ctime[1]
            mtime_ns = entry.mtime[0] * 10**9 + entry.mtime[1]

            if (stat.st_ctime_ns != ctime_ns) or (stat.st_mtime_ns != mtime_ns):
                with open(full_path, "rb") as fd:
                    new_sha = object_hash(fd, b"blob", None)
                    if not (entry.sha == new_sha):
                        print("    modified:", entry.name)

        if entry.name in all_files:
            all_files.remove(entry.name)
    
    print("\nUntracked files: ")
    for f in all_files:
        if not check_ignore(ignore, f):
            print(" ", f)

def index_write(repo, index):
    endianness = "big"
    with open(repo_file(repo, "index"), "wb") as f:
        # Write the magic bytes
        f.write(b"DIRC")
        # Write the version number
        f.write(index.version.to_bytes(4, endianness))
        # Write the number of entries
        f.write(len(index.entries).to_bytes(4, endianness))

        idx = 0
        for e in index.entries:
            f.write(e.ctime[0].to_bytes(4, endianness))
            f.write(e.ctime[1].to_bytes(4, endianness))
            f.write(e.mtime[0].to_bytes(4, endianness))
            f.write(e.mtime[1].to_bytes(4, endianness))
            f.write(e.dev.to_bytes(4, endianness))
            f.write(e.ino.to_bytes(4, endianness))

            # Mode
            mode = (e.mode_type << 12) | e.mode_perms
            f.write(mode.to_bytes(4, endianness))

            f.write(e.uid.to_bytes(4, endianness))
            f.write(e.gid.to_bytes(4, endianness))

            f.write(e.fsize.to_bytes(4, endianness))
            f.write(int(e.sha, 16).to_bytes(20, endianness))

            flag_assume_valid = 0x1 << 15 if e.flag_assume_valid else 0 # TODO

            name_bytes = e.name.encode("utf8")
            bytes_len = len(name_bytes)
            if bytes_len >= 0xFFF:
                name_length = 0xFFF
            else:
                name_length = bytes_len

            # Merge back three pieces of data (two flags and the
            # length of the name) on the same two bytes.
            f.write((flag_assume_valid | e.flag_stage | name_length).to_bytes(2, endianness))

            f.write(name_bytes)
            f.write((0).to_bytes(1, endianness))

            idx += 62 + len(name_bytes) + 1

            # Add padding if necessary
            if idx % 8 != 0:
                pad = 8 - (idx % 8)
                f.write((0).to_bytes(pad, endianness))
                idx += pad


def rm(repo, paths, delete=True, skip_missing=False):
    index = index_read(repo)

    worktree = repo.worktree + os.sep

    # Make paths absolute
    abspaths = list()
    for path in paths:
        abspath = os.path.abspath(path)
        if abspath.startswith(worktree):
            abspaths.append(abspath)
        else:
            raise Exception("Cannot remove path outside of repo : {}".format(paths))
    
    kept_entries = list()
    remove = list()

    for e in index.entries:
        full_path = os.path.join(repo.worktree, e.name)

        if full_path in abspaths:
            remove.append(full_path)
            abspaths.remove(full_path)
        else:
            kept_entries.append(e)
    
    if len(abspaths) > 0 and not skip_missing:
        raise Exception("Cannot remove paths not in the index {}".format(abspaths))

    if delete:
        for path in remove:
            os.unlink(path)

    index.entries = kept_entries;
    index_write(repo, index)


def add(repo, paths, delete=True, skip_missing=False):

    # First remove all the files from the index
    rm(repo, paths, delete=False, skip_missing=True)

    worktree = repo.worktree + os.sep

    # Convert the paths to pairs: (absolute, relative_to_worktree)
    clean_paths = list()
    for path in paths:
        abspath = os.path.abspath(path)
        if not (abspath.startswith(worktree) and os.path.isfile(abspath)):
            raise Exception("Not a file, or outside the repo: {}".format(paths))
        relpath = os.path.relpath(abspath, repo.worktree)
        clean_paths.append((abspath, relpath))
    
    index = index_read(repo)

    for (abspath, relpath) in clean_paths:
        with open(abspath, "rb") as fd:
            sha = object_hash(fd, b"blob", repo)

        stat = os.stat(abspath)
        ctime_s = int(stat.st_ctime)
        ctime_ns = stat.st_ctime_ns % 10**9 # TODO
        mtime_s = int(stat.st_mtime)
        mtime_ns = stat.st_mtime_ns % 10**9

        entry = GitIndexEntry(ctime=(ctime_s, ctime_ns), mtime=(mtime_s, mtime_ns), dev=stat.st_dev, ino=stat.st_ino,
                    mode_type=0b1000, mode_perms=0o644, uid=stat.st_uid, gid=stat.st_gid,
                    fsize=stat.st_size, sha=sha, flag_assume_valid=False,
                    flag_stage=False, name=relpath)

        index.entries.append(entry)
    
    index_write(repo, index)


def gitconfig_read():
    xdg_config_home = os.environ["XDG_CONFIG_HOME"] if "XDG_CONFIG_HOME" in os.environ else "~/.config"
    configfiles = [
        os.path.expanduser(os.path.join(xdg_config_home, "git/config")),
        os.path.expanduser("~/..gitconfig")
    ] # TODO

    config = configparser.ConfigParser()
    config.read(configfiles)
    return config

def gitconfig_user_get(config):
    if "user" in config:
        if "name" in config["user"] and "email" in config["user"]:
            return "{} <{}>".format(config["user"]["name"], config["user"]["email"])
    return None


def tree_from_index(repo, index):
    contents = dict()
    contents[""] = list() # root folder

    # Enumerate entries and turn then into a dictionary where keys
    # are directories and values are lists of directory contents
    for entry in index.entries:
        dirname = os.path.dirname(entry.name)

        key = dirname
        while key != "":
            if not key in contents:
                contents[key] = list()
            key = os.path.dirname(key)

        contents[dirname].append(entry)

    # Get keys (directories) and sort them by length, descending
    # We always want encounter a child directory before its parent
    sorted_paths = sorted(contents.keys(), key=len, reverse=True)
    
    # Conatins SHA of the current tree
    sha = None

    for path in sorted_paths:
        tree = GitTree()

        for entry in contents[path]:
            if isinstance(entry, GitIndexEntry):
                leaf_mode = "{:02o}{:04o}".format(entry.mode_type, entry.mode_perms).encode("ascii") # TODO
                leaf = GitTreeLeaf(mode=leaf_mode, path=os.path.basename(entry.name), sha=entry.sha)
            else:
                leaf = GitTreeLeaf(mode=b"040000", path=entry[0], sha=entry[1])
            
            tree.items.append(leaf)
        
        sha = object_write(tree, repo)

        parent = os.path.dirname(path)
        base = os.path.basename(path)
        contents[parent].append((base, sha))

    return sha

def commit_create(repo, tree, parent, author, timestamp, message):
    if author == None:
        author = "Vishal Salgond" # TODO

    commit = GitCommit()
    commit.kvlm[b"tree"] = tree.encode("ascii")
    if parent:
        commit.kvlm[b"parent"] = parent.encode("ascii")

    offset = int(timestamp.astimezone().utcoffset().total_seconds())
    hours = offset // 3600
    minutes = (offset % 3600) // 60
    tz = "{}{:02}{:02}".format("+" if offset > 0 else "-", hours, minutes)

    author = author + timestamp.strftime(" %s ") + tz

    commit.kvlm[b"author"] = author.encode("utf8")
    commit.kvlm[b"committer"] = author.encode("utf8")
    commit.kvlm[None] = message.encode("utf8")

    return object_write(commit, repo)


# Class definitions

class GitRepository(object):

    worktree = None
    gitdir = None
    conf = None

    def __init__(self, path, force=False):
        self.worktree = path
        self.gitdir = os.path.join(path, ".git")

        if not (force or os.path.isdir(self.gitdir)):
            raise Exception("[ERROR]: {} is not a git repository.".format(path))

        self.conf = configparser.ConfigParser()
        cf = repo_file(self, "config")

        if cf and os.path.exists(cf):
            self.conf.read([cf])
        elif not force:
            raise Exception("[ERROR]: Configuration file is missing.")

        if not force:
            ver = int(self.conf.get("core", "repositoryformatversion"))
            if ver != 0:
                raise Exception("[ERROR]: The repositoryformatversion {} is not supported.".format(ver))

class GitObject(object):
    
    def __init__(self, data=None):
        if data != None:
            self.deserialize(data)
        else:
            self.init()
    
    """Below functions MUST be implemented by subclasses."""
    def serialize(self, repo):
        raise Exception("[ERROR]: Unexpected error occured.")
    
    def deserialize(self, data):
        raise Exception("[ERROR]: Unexpected error occured.")

class GitCommit(GitObject):
    fmt = b'commit'

    def serialize(self):
        return kvlm_serialize(self.kvlm)

    def deserialize(self, data):
        self.kvlm = kvlm_parse(data)

    def init(self):
        self.kvlm = dict()

class GitTree(GitObject):
    fmt = b'tree'

    def serialize(self):
        return tree_serialize(self)
    
    def deserialize(self, data):
        self.items = tree_parse(data)

    def init(self):
        self.items = list()


class GitBlob(GitObject):
    fmt = b'blob'

    def serialize(self):
        return self.blobdata

    def deserialize(self, data):
        self.blobdata = data

class GitTag(GitCommit):
    fmt = b'tag'

class GitTreeLeaf(object):
    def __init__(self, mode, path, sha):
        self.mode = mode
        self.path = path
        self.sha = sha

class GitIndexEntry(object):
    def __init__(self, ctime=None, mtime=None, dev=None, ino=None, mode_type=None,
                mode_perms=None, uid=None, gid=None, fsize=None, sha=None, flag_assume_valid=None,
                flag_stage=None, name=None):
        
        # The last time a file's metadata changed, this is pair - (timestamp in sec, nanosec)
        self.ctime = ctime

        #The last time a file's data changed. 
        self.mtime = mtime

        # The ID of the devive containing this file
        self.dev = dev

        # The file's inode number
        self.ino = ino

        # Object type (b1000 - regular, b1010 - symlink, b110 - gitlink)
        self.mode_type = mode_type

        # Object permissions in integer
        self.mode_perms = mode_perms

        # User ID of the owner
        self.uid = uid

        # Group ID of the owner
        self.gid = gid

        # Size of this object, in bytes
        self.fsize = fsize

        self.sha = sha
        self.flag_assume_valid = flag_assume_valid
        self.flag_stage = flag_stage
        self.name = name

class GitIndex(object):
    version = None
    entries = []

    def __init__(self, version=2, entries=None):
        if not entries:
            entries = list()
        
        self.version = version
        self.entries = entries


class GitIgnore(object):
    absolute = None
    scoped = None

    def __init__(self, absolute, scoped):
        self.absolute = absolute
        self.scoped = scoped

#############################################################
# wyag init
# usage: wyag init <path>
#############################################################

def cmd_init(args):
    repo_create(args.path)

#############################################################
# wyag cat-file
# usage: wyag cat-file TYPE OBJECT
#############################################################

def cmd_cat_file(args):
    repo = repo_find()
    cat_file(repo, args.object, fmt=args.type.encode())

#############################################################
# wyag cat-file
# usage: wyag cat-file TYPE OBJECT
#############################################################

def cmd_hash_object(args):
    if args.write:
        repo = repo_find()
    else:
        repo = None
    
    with open(args.path, "rb") as fd:
        sha = object_hash(fd, args.type.encode(), repo)
        print(sha)

#############################################################
# wyag log
# usage: wyag log <commit>
#############################################################
    
def cmd_log(args):
    repo = repo_find()

    print("digraph wyaglog{")
    print("  node[shape=rect]")
    log_graphviz(repo, object_find(repo, args.commit), set())
    print("}")


#############################################################
# wyag ls-tree
# usage: wyag ls-tree [-r] TREE
#############################################################

def cmf_ls_tree(args):
    repo = repo_find()
    ls_tree(repo, args.tree, args.recursive)


#############################################################
# wyag checkout
# usage: wyag checkout <commit> <path>
#############################################################

def cmd_checkout(args):
    repo = repo_find()
    obj = object_read(repo, object_find(repo, args.commit))

    # If the object is a commit, grab its tree
    if obj.fmt == b'commit':
        obj = object_read(repo, obj.kvlm[b'tree'].decode("ascii"))

    # Verify that the given path is an empty directory
    if os.path.exists(args.path):
        if not os.path.isdir(args.path):
            raise Exception("Not a directory: {0}".format(args.path))
        if os.listdir(args.path):
            raise Exception("Not an empty directory: {0}".format(args.path))
    else:
        os.makedirs(args.path)
    
    tree_checkout(repo, obj, os.path.realpath(args.path))


#############################################################
# wyag show-ref
# usage: wyag show-ref
#############################################################

def cmd_show_ref(args):
    repo = repo_find()
    refs = ref_list(repo)
    show_ref(repo, refs, prefix="refs")


#############################################################
# wyag tag
# usage: wyag tag [-a] NAME [OBJECT]
#############################################################

def cmd_tag(args):
    repo = repo_find()

    if args.name:
        tag_create(repo, args.name, args.object,
            type="object" if args.create_tag_object else "ref")
    else:
        refs = ref_list(repo)
        show_ref(repo, refs["tags"], with_hash=False)


#############################################################
# wyag rev-parse 
# usage: wyag rev-parse --wyag-type [TYPE] [NAME]
#############################################################

def cmd_rev_parse(args):
    fmt = None

    if args.type:
        fmt = args.type.encode()
    
    repo = repo_find()
    print(object_find(repo, args.name, fmt, follow=True))


#############################################################
# wyag ls-files
# usage: wyag ls-files [--verbose]
#############################################################

def cmd_ls_files(args):
    repo = repo_find()
    index = index_read(repo)
    if args.verbose:
        print("Index file with v{} containing {} entries.".format(index.version, index.entries))
    
    mode_dict = { 0b1000: "regular file",
                0b1010: "symlink",
                0b1110: "git link" }

    for e in index.entries:
        print(e.name)
        print("    {} with perms {:o}".format(mode_dict[e.mode_type], e.mode_perms))
        print("    on blob: {}".format(e.sha))
        print("    created {}.{}, modified {}.{}".format(
            datetime.fromtimestamp(e.ctime[0]),
            e.ctime[1],
            datetime.fromtimestamp(e.mtime[0]),
            e.mtime[1]
        ))
        print("    device {}, inode {}".format(e.dev, e.ino))
        print("    user: {} ({}), group: {} ({})".format(
            pwd.getpwuid(e.uid).pw_name,
            e.uid,
            grp.getgrgid(e.gid).gr_name,
            e.gid
        ))
        print("    flags: stage={} assume_valid={}".format(
            e.flag_stage,
            e.flag_assume_valid
        ))

#############################################################
# wyag check-ignore
# usage: wyag check-ignore [path]
#############################################################

def cmd_check_ignore(args):
    repo = repo_find()
    rules = gitignore_read(repo)
    for path in args.path:
        if check_ignore(rules, path):
            print(path)


#############################################################
# wyag check-ignore
# usage: wyag check-ignore [path]
#############################################################

def cmd_status(_):
    repo = repo_find()
    index = index_read(repo)

    cmd_status_branch(repo)
    cmd_status_head_index(repo, index)
    print() # new line
    cmd_status_index_worktree(repo, index)

#############################################################
# wyag rm
# usage: wyag rm [path]
#############################################################

def cmd_rm(args):
  repo = repo_find()
  rm(repo, args.path)

#############################################################
# wyag add
# usage: wyag add [path]
#############################################################

def cmd_add(args):
  repo = repo_find()
  add(repo, args.path)


#############################################################
# wyag commit
# usage: wyag commit -m [message]
#############################################################

def cmd_commit(args):
    repo = repo_find()
    index = index_read(repo)

    tree = tree_from_index(repo, index)
    commit = commit_create(repo, tree, object_find(repo, "HEAD"),
        gitconfig_user_get(gitconfig_read()), datetime.now(), args.message)
    
    # Update HEAD so out commit is now the tip of the active branch
    active_branch = branch_get_active(repo)
    if active_branch:
        with open(repo_file(repo, os.path.join("refs/heads", active_branch)), "w") as fd:
            fd.write(commit + "\n")
    else:
        with open(repo_file(repo, "HEAD"), "w") as fd:
            fd.write("\n")