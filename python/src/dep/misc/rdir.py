import os
import datetime
import fnmatch

def rdir(rootdir='*', *args):
    def recursive_list(prepath, wildpath, postpath):
        if not wildpath:
            D = [os.path.join(prepath, f) for f in os.listdir(prepath) if not isdotdir(f) and not issvndir(f)]
            if os.path.isdir(prepath):
                fullpath = prepath
            else:
                fullpath = os.path.dirname(prepath)
            return sorted(D, key=lambda x: (not os.path.isdir(x), x))
        elif wildpath == '**':
            D = recursive_list(prepath, '', postpath[1:])
            D_sd = [os.path.join(prepath, f) for f in os.listdir(prepath) if not isdotdir(f) and not issvndir(f) and os.path.isdir(os.path.join(prepath, f))]
            for sd in D_sd:
                D.extend(recursive_list(sd + os.sep, '**', postpath))
            return D
        else:
            D_sd = [os.path.join(prepath, f) for f in fnmatch.filter(os.listdir(prepath), wildpath) if not isdotdir(f) and not issvndir(f) and os.path.isdir(os.path.join(prepath, f))]
            result = []
            for sd in D_sd:
                result.extend(recursive_list(sd + os.sep, '', postpath))
            return result

    prepath, wildpath, postpath = '', '', rootdir
    sep_pos = rootdir.rfind(os.sep)
    if sep_pos != -1:
        prepath = rootdir[:sep_pos + 1]
        postpath = rootdir[sep_pos + 1:]
        star_pos = prepath.find('*')
        if star_pos != -1:
            postpath = prepath[star_pos:] + postpath
            prepath = prepath[:star_pos]
            sep_pos = prepath.rfind(os.sep)
            if sep_pos != -1:
                wildpath = prepath[sep_pos + 1:]
                prepath = prepath[:sep_pos + 1]
            sep_pos = postpath.find(os.sep)
            if sep_pos != -1:
                wildpath += postpath[:sep_pos]
                postpath = postpath[sep_pos:]

    D = recursive_list(prepath, wildpath, postpath)

    if len(args) >= 1 and args[0]:
        test_expr = args[0]
        if callable(test_expr):
            D = [d for d in D if test_expr(d)]
        else:
            raise ValueError(f'Invalid TEST "{test_expr}"')

    common_path = ''
    if len(args) >= 2 and args[1]:
        arg2 = args[1]
        if isinstance(arg2, str):
            common_path = arg2
        elif isinstance(arg2, (int, bool)) and arg2:
            common_path = prepath
        rm_path = common_path
        is_common = all(d.startswith(rm_path) for d in D)
        if is_common:
            D = [d.replace(rm_path, '') for d in D]
        else:
            common_path = ''

    if len(args) >= 3:
        if args[2]:
            if len(D) == 0:
                print(f'{rootdir} not found.')
            else:
                for d in D:
                    if os.path.isdir(d):
                        print(f'{"":>29} {d:<64}')
                    else:
                        sz = os.path.getsize(d)
                        ss = min(4, (sz.bit_length() - 1) // 10) if sz > 0 else 0
                        print(f'{sz / 1024**ss:4.0f} {"kMGTP"[ss]}b  {os.path.getmtime(d):20.0f}  {d:<64}')

    return D if common_path == '' else (D, common_path)

def issvndir(d):
    """
    True for ".svn" directories.
    d is a list of os.DirEntry objects returned by os.scandir() or similar function.
    """
    is_dir = [entry.is_dir() for entry in d]
    is_svn = [entry.name == '.svn' for entry in d]

    # Uncomment the following line to disable ".svn" filtering
    # is_svn = [False] * len(d)

    tf = [dir_ and svn for dir_, svn in zip(is_dir, is_svn)]

    return tf

def isdotdir(d):
    """
    True for "." and ".." directories.
    d is a list of os.DirEntry objects returned by os.scandir() or similar function.
    """
    is_dir = [entry.is_dir() for entry in d]
    is_dot = [entry.name == '.' for entry in d]
    is_dotdot = [entry.name == '..' for entry in d]

    tf = [dir_ and (dot or dotdot) for dir_, dot, dotdot in zip(is_dir, is_dot, is_dotdot)]
    
    return tf

def evaluate(d, expr):
    """
    True for item where evaluated expression is correct or return a non empty cell.
    d is a list of dictionaries returned by os.scandir() or similar function.
    """
    # Get fields that can be used
    name = [entry.name for entry in d]
    date = [datetime.datetime.fromtimestamp(entry.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S') for entry in d]
    datenum = [entry.stat().st_mtime for entry in d]
    bytes_ = [entry.stat().st_size for entry in d]
    isdir = [entry.is_dir() for entry in d]

    # Create a local scope dictionary for eval
    local_scope = {
        'name': name,
        'date': date,
        'datenum': datenum,
        'bytes': bytes_,
        'isdir': isdir
    }

    tf = eval(expr, {}, local_scope)

    # Convert list outputs returned by filters to a logical
    if isinstance(tf, list):
        tf = [not bool(x) for x in tf]

    return tf
