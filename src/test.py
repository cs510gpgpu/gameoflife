#!/usr/bin/env python

import os
import platform
import subprocess
import json
import datetime

def exe(name):
    p = platform.system().lower()
    f = {'linux':lambda s: "./" + s,
         'windows':lambda s: s + ".exe"}
    return f[p](name)

def sep():
    p = platform.system().lower()
    f = {'linux':'/',
         'windows':'\\'}
    return f[p]

def compile(cfg, target, blocks=None):
    if not blocks:
        blocks = 32

    p = platform.system().lower()
    cmds = cfg['cmds']['platform'][p]
    defines = {"TILE_WIDTH":blocks}
    plat_defines = " ".join(map(lambda e : cmds['define'].format(name=e[0], value=e[1]), iter(defines.items())))
    build_cmd = cmds['build'].format(defines=plat_defines, target=target['name'])
    subprocess.call(cmds['clean'], shell=True)
    print(build_cmd)
    subprocess.call(build_cmd, shell=True)
    
    for (width, height) in map(lambda r: (r[0], r[1]), cfg['resolutions']):
        for mode in target['modes']:
            for prof in cfg['cmds']['prof']:
                args = {'binary':target['name'], 'block_dim':blocks, 'height':height, 'width':width, 'mode':mode}
                result = nvprof_dir + sep() + "_".join(map(str, args.values()))
                args['result'] = result
                args['binary'] = exe(args['binary'])
                prof_cmd = prof.format(**args)
                print (prof_cmd)
                subprocess.call(prof_cmd, shell=True)
        
def run_test(cfg, target):
    print("Running Tests for Target:{name}".format(**target))

    for block in target['blocks']:
        compile(cfg, target, block)


if __name__ == "__main__":
    d = datetime.datetime.now()
    nvprof_dir = "nvprof-{}".format(d.isoformat()).replace(":", ".")
    f = open("test.json")
    j = json.load(f)
    
    if not os.path.exists(nvprof_dir):
        os.makedirs(nvprof_dir)
            
    for t in j['targets']:
        run_test(j, t)
