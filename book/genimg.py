import re, os, io, hashlib
import matplotlib

matplotlib.rcParams['svg.hashsalt'] = 'reproducible'

os.chdir('src')

ns = {}
for tutorial in re.findall('[(](tutorial/.*[.]md)[)]', open('SUMMARY.md').read()):
    print('ENTERING', tutorial)
    for snippet in re.findall('^```python$(.*?)^```$', open(tutorial).read(), re.MULTILINE | re.DOTALL):
        print(snippet)
        try:
            exec(snippet, ns, ns)
        except Exception as e:
            print('ERROR:', e)
        else:
            for fignum in matplotlib.pyplot.get_fignums():
                with io.BytesIO() as f:
                    matplotlib.pyplot.figure(fignum).savefig(f, format='svg', metadata=dict(Date=None))
                    data = f.getvalue()
                figpath = f'{tutorial[:-3]}-{hashlib.sha1(data).hexdigest()[:8]}.svg'
                print('-->', figpath)
                with open(figpath, 'wb') as f:
                    f.write(data)
            matplotlib.pyplot.close('all')
