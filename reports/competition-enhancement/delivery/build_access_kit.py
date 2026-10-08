import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Build a hash-verified, explicitly selected jury archive.')
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--kit', choices=['view', 'current_model'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve(strict=True)
    target = args.output.resolve()
    if target == root or root in target.parents:
        raise ValueError('The archive must be outside the source repository')
    if target.exists():
        raise FileExistsError(target)
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    files = selection['kits'][args.kit]['files']
    checked = {}
    for relative, meta in files.items():
        path = root / relative
        resolved = path.resolve(strict=True)
        if root not in resolved.parents or path.is_symlink():
            raise ValueError('Unsafe source path: '+relative)
        content = path.read_bytes()
        if len(content) != meta['bytes'] or digest(content) != meta['sha256']:
            raise ValueError('Source changed after the selection was prepared: '+relative)
        checked[relative] = content
    target.parent.mkdir(parents=True, exist_ok=True)
    archive_manifest = {'base_commit':selection['base_commit'], 'kit':args.kit,
                        'scope':selection['kits'][args.kit]['scope'], 'files':files,
                        'publication_status':'LOCAL_ONLY; not published or submitted'}
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr('FILE-SHA256.json', json.dumps(archive_manifest,ensure_ascii=False,indent=2)+'\n')
        for relative, content in checked.items():
            archive.writestr(relative, content)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC mismatch')
        if set(archive.namelist()) != set(files)|{'FILE-SHA256.json'}:
            raise ValueError('Unexpected archive members')
        for relative, meta in files.items():
            if digest(archive.read(relative)) != meta['sha256']:
                raise ValueError('Archive SHA mismatch: '+relative)
    proof = {'status':'PASS', 'base_commit':selection['base_commit'], 'kit':args.kit,
             'archive':str(target), 'bytes':target.stat().st_size,
             'sha256':digest(target.read_bytes()), 'member_count':len(files)+1,
             'crc':True, 'all_member_sha256':True,
             'scientific_commands_rerun':False, 'published':False}
    target.with_suffix(target.suffix+'.verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
