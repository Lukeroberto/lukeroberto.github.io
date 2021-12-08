
```python
%load_ext autoreload
%autoreload 2
!pip install -r requirements.txt

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import clear_output, Image


import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

from envs.bandit import Bandit
from envs.context_bandit import ContextualBandit, get_state_probs
from envs.frozen_lake import FrozenLakeEnv

from utils import *
from value_iteration import *
from mcts import *

```

    Collecting jupyter
      Using cached jupyter-1.0.0-py2.py3-none-any.whl (2.7 kB)
    Collecting ipywidgets
      Downloading ipywidgets-7.6.5-py2.py3-none-any.whl (121 kB)
    Collecting gym
      Downloading gym-0.21.0.tar.gz (1.5 MB)
    Requirement already satisfied: numpy in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from -r requirements.txt (line 4)) (1.19.5)
    Collecting matplotlib==3.1.0
      Downloading matplotlib-3.1.0.tar.gz (37.2 MB)
    Requirement already satisfied: seaborn in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from -r requirements.txt (line 6)) (0.11.1)
    Requirement already satisfied: scipy in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from -r requirements.txt (line 7)) (1.6.2)
    Collecting imageio
      Downloading imageio-2.13.3-py3-none-any.whl (3.3 MB)
    Requirement already satisfied: cycler>=0.10 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from matplotlib==3.1.0->-r requirements.txt (line 5)) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from matplotlib==3.1.0->-r requirements.txt (line 5)) (1.3.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from matplotlib==3.1.0->-r requirements.txt (line 5)) (2.4.7)
    Requirement already satisfied: python-dateutil>=2.1 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from matplotlib==3.1.0->-r requirements.txt (line 5)) (2.8.1)
    Requirement already satisfied: six in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from cycler>=0.10->matplotlib==3.1.0->-r requirements.txt (line 5)) (1.15.0)
    Collecting cloudpickle>=1.2.0
      Downloading cloudpickle-2.0.0-py3-none-any.whl (25 kB)
    Collecting pillow>=8.3.2
      Downloading Pillow-8.4.0-cp38-cp38-win_amd64.whl (3.2 MB)
    Collecting nbformat>=4.2.0
      Downloading nbformat-5.1.3-py3-none-any.whl (178 kB)
    Requirement already satisfied: ipython-genutils~=0.2.0 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from ipywidgets->-r requirements.txt (line 2)) (0.2.0)
    Requirement already satisfied: ipython>=4.0.0 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from ipywidgets->-r requirements.txt (line 2)) (7.22.0)
    Requirement already satisfied: traitlets>=4.3.1 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from ipywidgets->-r requirements.txt (line 2)) (5.0.5)
    Requirement already satisfied: ipykernel>=4.5.1 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from ipywidgets->-r requirements.txt (line 2)) (5.5.3)
    Collecting widgetsnbextension~=3.5.0
      Downloading widgetsnbextension-3.5.2-py2.py3-none-any.whl (1.6 MB)
    Collecting jupyterlab-widgets>=1.0.0
      Downloading jupyterlab_widgets-1.0.2-py3-none-any.whl (243 kB)
    Requirement already satisfied: tornado>=4.2 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from ipykernel>=4.5.1->ipywidgets->-r requirements.txt (line 2)) (6.1)
    Requirement already satisfied: jupyter-client in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from ipykernel>=4.5.1->ipywidgets->-r requirements.txt (line 2)) (6.1.12)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from ipython>=4.0.0->ipywidgets->-r requirements.txt (line 2)) (3.0.18)
    Requirement already satisfied: backcall in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from ipython>=4.0.0->ipywidgets->-r requirements.txt (line 2)) (0.2.0)
    Requirement already satisfied: decorator in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from ipython>=4.0.0->ipywidgets->-r requirements.txt (line 2)) (5.0.6)
    Requirement already satisfied: pickleshare in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from ipython>=4.0.0->ipywidgets->-r requirements.txt (line 2)) (0.7.5)
    Requirement already satisfied: setuptools>=18.5 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from ipython>=4.0.0->ipywidgets->-r requirements.txt (line 2)) (52.0.0.post20210125)
    Requirement already satisfied: colorama in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from ipython>=4.0.0->ipywidgets->-r requirements.txt (line 2)) (0.4.4)
    Requirement already satisfied: pygments in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from ipython>=4.0.0->ipywidgets->-r requirements.txt (line 2)) (2.8.1)
    Requirement already satisfied: jedi>=0.16 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from ipython>=4.0.0->ipywidgets->-r requirements.txt (line 2)) (0.18.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from jedi>=0.16->ipython>=4.0.0->ipywidgets->-r requirements.txt (line 2)) (0.8.2)
    Requirement already satisfied: jupyter-core in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from nbformat>=4.2.0->ipywidgets->-r requirements.txt (line 2)) (4.7.1)
    Collecting jsonschema!=2.5.0,>=2.4
      Downloading jsonschema-4.2.1-py3-none-any.whl (69 kB)
    Collecting pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0
      Downloading pyrsistent-0.18.0-cp38-cp38-win_amd64.whl (62 kB)
    Collecting attrs>=17.4.0
      Downloading attrs-21.2.0-py2.py3-none-any.whl (53 kB)
    Collecting importlib-resources>=1.4.0
      Downloading importlib_resources-5.4.0-py3-none-any.whl (28 kB)
    Collecting zipp>=3.1.0
      Downloading zipp-3.6.0-py3-none-any.whl (5.3 kB)
    Requirement already satisfied: wcwidth in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=4.0.0->ipywidgets->-r requirements.txt (line 2)) (0.2.5)
    Collecting notebook>=4.4.1
      Downloading notebook-6.4.6-py3-none-any.whl (9.9 MB)
    Collecting nest-asyncio>=1.5
      Downloading nest_asyncio-1.5.4-py3-none-any.whl (5.1 kB)
    Collecting terminado>=0.8.3
      Downloading terminado-0.12.1-py3-none-any.whl (15 kB)
    Requirement already satisfied: pyzmq>=17 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->-r requirements.txt (line 2)) (22.0.3)
    Collecting jinja2
      Downloading Jinja2-3.0.3-py3-none-any.whl (133 kB)
    Collecting Send2Trash>=1.8.0
      Downloading Send2Trash-1.8.0-py3-none-any.whl (18 kB)
    Collecting prometheus-client
      Downloading prometheus_client-0.12.0-py2.py3-none-any.whl (57 kB)
    Collecting nbconvert
      Downloading nbconvert-6.3.0-py3-none-any.whl (556 kB)
    Collecting argon2-cffi
      Downloading argon2_cffi-21.2.0-py3-none-any.whl (14 kB)
    Requirement already satisfied: pywin32>=1.0 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from jupyter-core->nbformat>=4.2.0->ipywidgets->-r requirements.txt (line 2)) (300)
    Collecting pywinpty>=1.1.0
      Downloading pywinpty-1.1.6-cp38-none-win_amd64.whl (1.4 MB)
    Collecting qtconsole
      Downloading qtconsole-5.2.1-py3-none-any.whl (120 kB)
    Collecting jupyter-console
      Downloading jupyter_console-6.4.0-py3-none-any.whl (22 kB)
    Requirement already satisfied: pandas>=0.23 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from seaborn->-r requirements.txt (line 6)) (1.2.4)
    Requirement already satisfied: pytz>=2017.3 in c:\users\luke1\miniconda3\envs\m37\lib\site-packages (from pandas>=0.23->seaborn->-r requirements.txt (line 6)) (2021.1)
    Collecting argon2-cffi-bindings
      Downloading argon2_cffi_bindings-21.2.0-cp36-abi3-win_amd64.whl (30 kB)
    Collecting cffi>=1.0.1
      Downloading cffi-1.15.0-cp38-cp38-win_amd64.whl (179 kB)
    Collecting pycparser
      Downloading pycparser-2.21-py2.py3-none-any.whl (118 kB)
    Collecting MarkupSafe>=2.0
      Downloading MarkupSafe-2.0.1-cp38-cp38-win_amd64.whl (14 kB)
    Collecting nbclient<0.6.0,>=0.5.0
      Downloading nbclient-0.5.9-py3-none-any.whl (69 kB)
    Collecting bleach
      Downloading bleach-4.1.0-py2.py3-none-any.whl (157 kB)
    Collecting pandocfilters>=1.4.1
      Downloading pandocfilters-1.5.0-py2.py3-none-any.whl (8.7 kB)
    Collecting jupyterlab-pygments
      Using cached jupyterlab_pygments-0.1.2-py2.py3-none-any.whl (4.6 kB)
    Collecting defusedxml
      Downloading defusedxml-0.7.1-py2.py3-none-any.whl (25 kB)
    Collecting entrypoints>=0.2.2
      Using cached entrypoints-0.3-py2.py3-none-any.whl (11 kB)
    Collecting mistune<2,>=0.8.1
      Using cached mistune-0.8.4-py2.py3-none-any.whl (16 kB)
    Collecting testpath
      Downloading testpath-0.5.0-py3-none-any.whl (84 kB)
    Collecting webencodings
      Using cached webencodings-0.5.1-py2.py3-none-any.whl (11 kB)
    Collecting packaging
      Downloading packaging-21.3-py3-none-any.whl (40 kB)
    Collecting qtpy
      Downloading QtPy-1.11.3-py2.py3-none-any.whl (59 kB)
    Building wheels for collected packages: matplotlib, gym
      Building wheel for matplotlib (setup.py): started
      Building wheel for matplotlib (setup.py): finished with status 'error'
      Running setup.py clean for matplotlib
      Building wheel for gym (setup.py): started
      Building wheel for gym (setup.py): finished with status 'done'
      Created wheel for gym: filename=gym-0.21.0-py3-none-any.whl size=1616828 sha256=fb0c3af84fbf9014cd26d90f25919988ac9936938dbfe687574961d7f74084d2
      Stored in directory: c:\users\luke1\appdata\local\pip\cache\wheels\27\6d\b3\a3a6e10704795c9b9000f1ab2dc480dfe7bed42f5972806e73
    Successfully built gym
    Failed to build matplotlib
    Installing collected packages: zipp, pyrsistent, importlib-resources, attrs, pycparser, jsonschema, webencodings, packaging, nest-asyncio, nbformat, MarkupSafe, cffi, testpath, pywinpty, pandocfilters, nbclient, mistune, jupyterlab-pygments, jinja2, entrypoints, defusedxml, bleach, argon2-cffi-bindings, terminado, Send2Trash, prometheus-client, nbconvert, argon2-cffi, notebook, widgetsnbextension, qtpy, jupyterlab-widgets, qtconsole, pillow, matplotlib, jupyter-console, ipywidgets, cloudpickle, jupyter, imageio, gym
      Attempting uninstall: pillow
        Found existing installation: Pillow 8.1.2
        Uninstalling Pillow-8.1.2:
          Successfully uninstalled Pillow-8.1.2
      Attempting uninstall: matplotlib
        Found existing installation: matplotlib 3.4.1
        Uninstalling matplotlib-3.4.1:
          Successfully uninstalled matplotlib-3.4.1
        Running setup.py install for matplotlib: started
        Running setup.py install for matplotlib: finished with status 'error'
      Rolling back uninstall of matplotlib
      Moving to c:\users\luke1\miniconda3\envs\m37\lib\site-packages\__pycache__\pylab.cpython-38.pyc
       from C:\Users\luke1\AppData\Local\Temp\pip-uninstall-77k713n3\pylab.cpython-38.pyc
      Moving to c:\users\luke1\miniconda3\envs\m37\lib\site-packages\matplotlib-3.4.1-py3.8-nspkg.pth
       from C:\Users\luke1\AppData\Local\Temp\pip-uninstall-chqm5wiv\matplotlib-3.4.1-py3.8-nspkg.pth
      Moving to c:\users\luke1\miniconda3\envs\m37\lib\site-packages\matplotlib-3.4.1.dist-info\
       from C:\Users\luke1\Miniconda3\envs\m37\Lib\site-packages\~atplotlib-3.4.1.dist-info
      Moving to c:\users\luke1\miniconda3\envs\m37\lib\site-packages\matplotlib\
       from C:\Users\luke1\Miniconda3\envs\m37\Lib\site-packages\~atplotlib
      Moving to c:\users\luke1\miniconda3\envs\m37\lib\site-packages\mpl_toolkits\axes_grid1\
       from C:\Users\luke1\Miniconda3\envs\m37\Lib\site-packages\mpl_toolkits\~xes_grid1
      Moving to c:\users\luke1\miniconda3\envs\m37\lib\site-packages\mpl_toolkits\axes_grid\
       from C:\Users\luke1\Miniconda3\envs\m37\Lib\site-packages\mpl_toolkits\~xes_grid
      Moving to c:\users\luke1\miniconda3\envs\m37\lib\site-packages\mpl_toolkits\axisartist\
       from C:\Users\luke1\Miniconda3\envs\m37\Lib\site-packages\mpl_toolkits\~xisartist
      Moving to c:\users\luke1\miniconda3\envs\m37\lib\site-packages\mpl_toolkits\mplot3d\
       from C:\Users\luke1\Miniconda3\envs\m37\Lib\site-packages\mpl_toolkits\~plot3d
      Moving to c:\users\luke1\miniconda3\envs\m37\lib\site-packages\mpl_toolkits\tests\
       from C:\Users\luke1\Miniconda3\envs\m37\Lib\site-packages\mpl_toolkits\~ests
      Moving to c:\users\luke1\miniconda3\envs\m37\lib\site-packages\pylab.py
       from C:\Users\luke1\AppData\Local\Temp\pip-uninstall-chqm5wiv\pylab.py
    

      ERROR: Command errored out with exit status 1:
       command: 'C:\Users\luke1\Miniconda3\envs\m37\python.exe' -u -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'C:\\Users\\luke1\\AppData\\Local\\Temp\\pip-install-8238fwlv\\matplotlib_238ef439a6974f11be1e9134bb4d7c6f\\setup.py'"'"'; __file__='"'"'C:\\Users\\luke1\\AppData\\Local\\Temp\\pip-install-8238fwlv\\matplotlib_238ef439a6974f11be1e9134bb4d7c6f\\setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' bdist_wheel -d 'C:\Users\luke1\AppData\Local\Temp\pip-wheel-1wnit3z7'
           cwd: C:\Users\luke1\AppData\Local\Temp\pip-install-8238fwlv\matplotlib_238ef439a6974f11be1e9134bb4d7c6f\
      Complete output (499 lines):
      ================================================================================
      Edit setup.cfg to change the build options
      
      BUILDING MATPLOTLIB
        matplotlib: yes [3.1.0]
            python: yes [3.8.8 (default, Feb 24 2021, 15:54:32) [MSC v.1928 64 bit
                        (AMD64)]]
          platform: yes [win32]
      
      OPTIONAL SUBPACKAGES
       sample_data: yes [installing]
             tests: no  [skipping due to configuration]
      
      OPTIONAL BACKEND EXTENSIONS
               agg: yes [installing]
             tkagg: yes [installing; run-time loading from Python Tcl/Tk]
            macosx: no  [Mac OS-X only]
      
      OPTIONAL PACKAGE DATA
              dlls: no  [skipping due to configuration]
      
      running bdist_wheel
      running build
      running build_py
      creating build
      creating build\lib.win-amd64-3.8
      copying lib\pylab.py -> build\lib.win-amd64-3.8
      creating build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\afm.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\animation.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\artist.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\axis.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\backend_bases.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\backend_managers.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\backend_tools.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\bezier.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\blocking_input.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\category.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\cm.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\collections.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\colorbar.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\colors.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\container.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\contour.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\dates.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\docstring.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\dviread.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\figure.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\fontconfig_pattern.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\font_manager.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\gridspec.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\hatch.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\image.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\legend.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\legend_handler.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\lines.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\markers.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\mathtext.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\mlab.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\offsetbox.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\patches.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\path.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\patheffects.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\pylab.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\pyplot.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\quiver.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\rcsetup.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\sankey.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\scale.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\spines.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\stackplot.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\streamplot.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\table.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\texmanager.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\text.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\textpath.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\ticker.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\tight_bbox.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\tight_layout.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\transforms.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\type1font.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\units.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\widgets.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\_animation_data.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\_cm.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\_cm_listed.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\_color_data.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\_constrained_layout.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\_layoutbox.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\_mathtext_data.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\_pylab_helpers.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\_version.py -> build\lib.win-amd64-3.8\matplotlib
      copying lib\matplotlib\__init__.py -> build\lib.win-amd64-3.8\matplotlib
      creating build\lib.win-amd64-3.8\mpl_toolkits
      copying lib\mpl_toolkits\__init__.py -> build\lib.win-amd64-3.8\mpl_toolkits
      creating build\lib.win-amd64-3.8\matplotlib\axes
      copying lib\matplotlib\axes\_axes.py -> build\lib.win-amd64-3.8\matplotlib\axes
      copying lib\matplotlib\axes\_base.py -> build\lib.win-amd64-3.8\matplotlib\axes
      copying lib\matplotlib\axes\_secondary_axes.py -> build\lib.win-amd64-3.8\matplotlib\axes
      copying lib\matplotlib\axes\_subplots.py -> build\lib.win-amd64-3.8\matplotlib\axes
      copying lib\matplotlib\axes\__init__.py -> build\lib.win-amd64-3.8\matplotlib\axes
      creating build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_agg.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_cairo.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_gtk3.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_gtk3agg.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_gtk3cairo.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_macosx.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_mixed.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_nbagg.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_pdf.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_pgf.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_ps.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_qt4.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_qt4agg.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_qt4cairo.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_qt5.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_qt5agg.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_qt5cairo.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_svg.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_template.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_tkagg.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_tkcairo.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_webagg.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_webagg_core.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_wx.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_wxagg.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\backend_wxcairo.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\qt_compat.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\tkagg.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\windowing.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\wx_compat.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\_backend_pdf_ps.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\_backend_tk.py -> build\lib.win-amd64-3.8\matplotlib\backends
      copying lib\matplotlib\backends\__init__.py -> build\lib.win-amd64-3.8\matplotlib\backends
      creating build\lib.win-amd64-3.8\matplotlib\cbook
      copying lib\matplotlib\cbook\deprecation.py -> build\lib.win-amd64-3.8\matplotlib\cbook
      copying lib\matplotlib\cbook\__init__.py -> build\lib.win-amd64-3.8\matplotlib\cbook
      creating build\lib.win-amd64-3.8\matplotlib\compat
      copying lib\matplotlib\compat\subprocess.py -> build\lib.win-amd64-3.8\matplotlib\compat
      copying lib\matplotlib\compat\__init__.py -> build\lib.win-amd64-3.8\matplotlib\compat
      creating build\lib.win-amd64-3.8\matplotlib\projections
      copying lib\matplotlib\projections\geo.py -> build\lib.win-amd64-3.8\matplotlib\projections
      copying lib\matplotlib\projections\polar.py -> build\lib.win-amd64-3.8\matplotlib\projections
      copying lib\matplotlib\projections\__init__.py -> build\lib.win-amd64-3.8\matplotlib\projections
      creating build\lib.win-amd64-3.8\matplotlib\sphinxext
      copying lib\matplotlib\sphinxext\mathmpl.py -> build\lib.win-amd64-3.8\matplotlib\sphinxext
      copying lib\matplotlib\sphinxext\plot_directive.py -> build\lib.win-amd64-3.8\matplotlib\sphinxext
      copying lib\matplotlib\sphinxext\__init__.py -> build\lib.win-amd64-3.8\matplotlib\sphinxext
      creating build\lib.win-amd64-3.8\matplotlib\style
      copying lib\matplotlib\style\core.py -> build\lib.win-amd64-3.8\matplotlib\style
      copying lib\matplotlib\style\__init__.py -> build\lib.win-amd64-3.8\matplotlib\style
      creating build\lib.win-amd64-3.8\matplotlib\testing
      copying lib\matplotlib\testing\compare.py -> build\lib.win-amd64-3.8\matplotlib\testing
      copying lib\matplotlib\testing\conftest.py -> build\lib.win-amd64-3.8\matplotlib\testing
      copying lib\matplotlib\testing\decorators.py -> build\lib.win-amd64-3.8\matplotlib\testing
      copying lib\matplotlib\testing\determinism.py -> build\lib.win-amd64-3.8\matplotlib\testing
      copying lib\matplotlib\testing\disable_internet.py -> build\lib.win-amd64-3.8\matplotlib\testing
      copying lib\matplotlib\testing\exceptions.py -> build\lib.win-amd64-3.8\matplotlib\testing
      copying lib\matplotlib\testing\__init__.py -> build\lib.win-amd64-3.8\matplotlib\testing
      creating build\lib.win-amd64-3.8\matplotlib\tri
      copying lib\matplotlib\tri\triangulation.py -> build\lib.win-amd64-3.8\matplotlib\tri
      copying lib\matplotlib\tri\tricontour.py -> build\lib.win-amd64-3.8\matplotlib\tri
      copying lib\matplotlib\tri\trifinder.py -> build\lib.win-amd64-3.8\matplotlib\tri
      copying lib\matplotlib\tri\triinterpolate.py -> build\lib.win-amd64-3.8\matplotlib\tri
      copying lib\matplotlib\tri\tripcolor.py -> build\lib.win-amd64-3.8\matplotlib\tri
      copying lib\matplotlib\tri\triplot.py -> build\lib.win-amd64-3.8\matplotlib\tri
      copying lib\matplotlib\tri\trirefine.py -> build\lib.win-amd64-3.8\matplotlib\tri
      copying lib\matplotlib\tri\tritools.py -> build\lib.win-amd64-3.8\matplotlib\tri
      copying lib\matplotlib\tri\__init__.py -> build\lib.win-amd64-3.8\matplotlib\tri
      creating build\lib.win-amd64-3.8\matplotlib\backends\qt_editor
      copying lib\matplotlib\backends\qt_editor\figureoptions.py -> build\lib.win-amd64-3.8\matplotlib\backends\qt_editor
      copying lib\matplotlib\backends\qt_editor\formlayout.py -> build\lib.win-amd64-3.8\matplotlib\backends\qt_editor
      copying lib\matplotlib\backends\qt_editor\formsubplottool.py -> build\lib.win-amd64-3.8\matplotlib\backends\qt_editor
      copying lib\matplotlib\backends\qt_editor\_formlayout.py -> build\lib.win-amd64-3.8\matplotlib\backends\qt_editor
      copying lib\matplotlib\backends\qt_editor\__init__.py -> build\lib.win-amd64-3.8\matplotlib\backends\qt_editor
      creating build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
      copying lib\matplotlib\testing\jpl_units\Duration.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
      copying lib\matplotlib\testing\jpl_units\Epoch.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
      copying lib\matplotlib\testing\jpl_units\EpochConverter.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
      copying lib\matplotlib\testing\jpl_units\StrConverter.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
      copying lib\matplotlib\testing\jpl_units\UnitDbl.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
      copying lib\matplotlib\testing\jpl_units\UnitDblConverter.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
      copying lib\matplotlib\testing\jpl_units\UnitDblFormatter.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
      copying lib\matplotlib\testing\jpl_units\__init__.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
      creating build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\anchored_artists.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\angle_helper.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\axes_divider.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\axes_grid.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\axes_rgb.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\axes_size.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\axislines.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\axisline_style.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\axis_artist.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\clip_path.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\colorbar.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\floating_axes.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\grid_finder.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\grid_helper_curvelinear.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\inset_locator.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\parasite_axes.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      copying lib\mpl_toolkits\axes_grid\__init__.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
      creating build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
      copying lib\mpl_toolkits\axes_grid1\anchored_artists.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
      copying lib\mpl_toolkits\axes_grid1\axes_divider.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
      copying lib\mpl_toolkits\axes_grid1\axes_grid.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
      copying lib\mpl_toolkits\axes_grid1\axes_rgb.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
      copying lib\mpl_toolkits\axes_grid1\axes_size.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
      copying lib\mpl_toolkits\axes_grid1\colorbar.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
      copying lib\mpl_toolkits\axes_grid1\inset_locator.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
      copying lib\mpl_toolkits\axes_grid1\mpl_axes.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
      copying lib\mpl_toolkits\axes_grid1\parasite_axes.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
      copying lib\mpl_toolkits\axes_grid1\__init__.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
      creating build\lib.win-amd64-3.8\mpl_toolkits\axisartist
      copying lib\mpl_toolkits\axisartist\angle_helper.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
      copying lib\mpl_toolkits\axisartist\axes_divider.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
      copying lib\mpl_toolkits\axisartist\axes_grid.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
      copying lib\mpl_toolkits\axisartist\axes_rgb.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
      copying lib\mpl_toolkits\axisartist\axislines.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
      copying lib\mpl_toolkits\axisartist\axisline_style.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
      copying lib\mpl_toolkits\axisartist\axis_artist.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
      copying lib\mpl_toolkits\axisartist\clip_path.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
      copying lib\mpl_toolkits\axisartist\floating_axes.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
      copying lib\mpl_toolkits\axisartist\grid_finder.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
      copying lib\mpl_toolkits\axisartist\grid_helper_curvelinear.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
      copying lib\mpl_toolkits\axisartist\parasite_axes.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
      copying lib\mpl_toolkits\axisartist\__init__.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
      creating build\lib.win-amd64-3.8\mpl_toolkits\mplot3d
      copying lib\mpl_toolkits\mplot3d\art3d.py -> build\lib.win-amd64-3.8\mpl_toolkits\mplot3d
      copying lib\mpl_toolkits\mplot3d\axes3d.py -> build\lib.win-amd64-3.8\mpl_toolkits\mplot3d
      copying lib\mpl_toolkits\mplot3d\axis3d.py -> build\lib.win-amd64-3.8\mpl_toolkits\mplot3d
      copying lib\mpl_toolkits\mplot3d\proj3d.py -> build\lib.win-amd64-3.8\mpl_toolkits\mplot3d
      copying lib\mpl_toolkits\mplot3d\__init__.py -> build\lib.win-amd64-3.8\mpl_toolkits\mplot3d
      creating build\lib.win-amd64-3.8\matplotlib\mpl-data
      creating build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\sample_data\None_vs_nearest-pdf.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend
      copying lib\matplotlib\backends\web_backend\single_figure.html -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend
      creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.structure.min.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
      creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\images
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\images\ui-icons_cc0000_256x240.png -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\images
      creating build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts
      creating build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\afm\putri8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      creating build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizFourSymBol.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\afm\pncb8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\afm\pagdo8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      creating build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\filesave.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\back.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      creating build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\stylelib\seaborn-pastel.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\stylelib\seaborn.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\backends\web_backend\all_figures.html -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend
      copying lib\matplotlib\mpl-data\stylelib\fast.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\images\move.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\matplotlibrc -> build\lib.win-amd64-3.8\matplotlib\mpl-data
      copying lib\matplotlib\mpl-data\images\back.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXNonUni.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\images\qt4_editor_options.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\stylelib\seaborn-white.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\stylelib\seaborn-colorblind.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\sample_data\percent_bachelors_degrees_women_usa.csv -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\images\zoom_to_rect.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
      copying lib\matplotlib\mpl-data\images\filesave.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\zoom_to_rect.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\sample_data\ada.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\images\subplots.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\afm\phvlo8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      creating build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Symbol.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSerifDisplay.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\afm\pplbi8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXGeneralItalic.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\images\help.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\ttf\cmr10.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSerif-Bold.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\AUTHORS.txt -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
      copying lib\matplotlib\mpl-data\fonts\afm\cmtt10.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXNonUniBol.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\images\subplots.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\afm\pagk8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\stylelib\ggplot.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\sample_data\goog.npz -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\images\help.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\move.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\readme.txt -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      creating build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data\axes_grid
      copying lib\matplotlib\mpl-data\sample_data\axes_grid\bivariate_normal.npy -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data\axes_grid
      copying lib\matplotlib\mpl-data\images\home.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\css
      copying lib\matplotlib\backends\web_backend\css\fbm.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\css
      copying lib\matplotlib\mpl-data\images\home.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\afm\cmsy10.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\ttf\LICENSE_STIX -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\afm\pbkli8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\sample_data\msft.csv -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\stylelib\seaborn-poster.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\fonts\afm\phvb8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\stylelib\seaborn-talk.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\images\matplotlib.ppm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\subplots_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\sample_data\grace_hopper.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\fonts\afm\phvro8an.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\afm\pcrr8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\images\help_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\stylelib\seaborn-darkgrid.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Times-Italic.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      copying lib\matplotlib\mpl-data\fonts\afm\pagd8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\afm\pcrbo8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.theme.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
      copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Times-BoldItalic.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSans-BoldOblique.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\images\forward.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\forward_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\sample_data\embedding_in_wx3.xrc -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\images\ui-icons_777777_256x240.png -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\images
      copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Helvetica-BoldOblique.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSerif-BoldItalic.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\afm\ptmr8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\afm\pncr8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\js
      copying lib\matplotlib\backends\web_backend\js\mpl.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\js
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\index.html -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
      copying lib\matplotlib\mpl-data\sample_data\logo2.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\backends\web_backend\js\nbagg_mpl.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\js
      copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Courier-Bold.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      copying lib\matplotlib\mpl-data\fonts\afm\phvbo8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\stylelib\grayscale.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\stylelib\dark_background.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.min.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\images\ui-icons_555555_256x240.png -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\images
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizOneSymBol.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\images\back.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\afm\phvr8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\afm\cmmi10.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\afm\ptmb8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\afm\phvbo8an.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\afm\pncri8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\LICENSE.txt -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
      copying lib\matplotlib\mpl-data\images\filesave.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\home.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizFiveSymReg.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizFourSymReg.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\afm\phvb8an.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\ttf\cmmi10.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\images\matplotlib.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\matplotlib.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\sample_data\README.txt -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\images\filesave_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\backends\web_backend\js\mpl_tornado.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\js
      copying lib\matplotlib\mpl-data\fonts\afm\psyr.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\ZapfDingbats.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      copying lib\matplotlib\mpl-data\fonts\afm\ptmbi8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSerif.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\ttf\LICENSE_DEJAVU -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\images\hand.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\back_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Courier-BoldOblique.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\package.json -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
      copying lib\matplotlib\mpl-data\stylelib\seaborn-notebook.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\stylelib\seaborn-dark.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.structure.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
      copying lib\matplotlib\mpl-data\images\hand_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\move_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\sample_data\jacksboro_fault_dem.npz -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\images\back_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\qt4_editor_options.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\stylelib\tableau-colorblind10.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSans-Bold.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizThreeSymBol.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\images\forward.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Courier-Oblique.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSansMono-Oblique.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSerif-Italic.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\images\filesave_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\forward_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\move.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\sample_data\demodata.csv -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSansMono.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\afm\pplb8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\images\matplotlib_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizTwoSymReg.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\ttf\cmex10.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\images\zoom_to_rect.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSans-Oblique.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\stylelib\classic.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\stylelib\fivethirtyeight.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXNonUniIta.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\external
      creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\external\jquery
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\external\jquery\jquery.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\external\jquery
      copying lib\matplotlib\mpl-data\images\home.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\subplots.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\hand.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSansDisplay.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\stylelib\_classic_test.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\images\ui-icons_ffffff_256x240.png -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\images
      copying lib\matplotlib\mpl-data\stylelib\seaborn-paper.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\sample_data\eeg.dat -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizOneSymReg.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\images\home_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\move.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\afm\pzcmi8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\afm\pplri8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\images\zoom_to_rect_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\stylelib\seaborn-dark-palette.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXNonUniBolIta.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\images\hand.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\subplots_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery
      creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery\js
      copying lib\matplotlib\backends\web_backend\jquery\js\jquery.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery\js
      copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Helvetica-Oblique.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      copying lib\matplotlib\backends\web_backend\nbagg_uat.ipynb -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\images\ui-icons_444444_256x240.png -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\images
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXGeneral.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSansMono-Bold.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\afm\pbkl8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\images\move_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\home_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\stylelib\seaborn-ticks.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\images\forward.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\help.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Courier.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      copying lib\matplotlib\mpl-data\images\qt4_editor_options_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\matplotlib.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSans.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\sample_data\s1045.ima.gz -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\stylelib\seaborn-muted.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\fonts\afm\pcrro8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\sample_data\data_x_x2_x3.csv -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\fonts\afm\pbkd8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Helvetica-Bold.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      copying lib\matplotlib\mpl-data\images\help.ppm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\afm\putbi8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\backends\web_backend\css\page.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\css
      copying lib\matplotlib\mpl-data\images\help_large.ppm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizTwoSymBol.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\afm\putr8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\sample_data\ct.raw.gz -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSansMono-BoldOblique.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\images\zoom_to_rect_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
      copying lib\matplotlib\mpl-data\stylelib\seaborn-bright.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\sample_data\aapl.npz -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.theme.min.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
      copying lib\matplotlib\backends\web_backend\css\boilerplate.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\css
      copying lib\matplotlib\mpl-data\sample_data\Minduka_Present_Blue_Pack.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\fonts\afm\putb8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\ttf\cmss10.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\stylelib\seaborn-deep.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\fonts\afm\phvr8an.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\backends\web_backend\jquery\js\jquery.min.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery\js
      copying lib\matplotlib\mpl-data\fonts\afm\pncbi8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\afm\phvro8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\afm\pplr8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\ttf\cmb10.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\stylelib\seaborn-whitegrid.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\fonts\afm\pzdr.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.min.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
      copying lib\matplotlib\mpl-data\sample_data\grace_hopper.jpg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\fonts\afm\cmr10.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXGeneralBol.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizThreeSymReg.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\images\hand.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\subplots.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\images\zoom_to_rect.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\ttf\cmsy10.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\images\ui-icons_777620_256x240.png -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\images
      copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Times-Bold.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      copying lib\matplotlib\mpl-data\fonts\afm\ptmri8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\ttf\cmtt10.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\afm\pagko8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\afm\phvl8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\afm\pbkdi8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\fonts\afm\cmex10.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\sample_data\topobathy.npz -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\fonts\ttf\STIXGeneralBolIta.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
      copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Times-Roman.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      copying lib\matplotlib\backends\web_backend\ipython_inline_figure.html -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend
      copying lib\matplotlib\mpl-data\sample_data\membrane.dat -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
      copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Helvetica.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
      copying lib\matplotlib\mpl-data\images\back.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\stylelib\Solarize_Light2.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\images\qt4_editor_options.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\fonts\afm\pcrb8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
      copying lib\matplotlib\mpl-data\images\forward.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      copying lib\matplotlib\mpl-data\stylelib\bmh.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
      copying lib\matplotlib\mpl-data\images\filesave.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
      UPDATING build\lib.win-amd64-3.8\matplotlib\_version.py
      set build\lib.win-amd64-3.8\matplotlib\_version.py to '3.1.0'
      running build_ext
      building 'matplotlib.ft2font' extension
      error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
      ----------------------------------------
      ERROR: Failed building wheel for matplotlib
        ERROR: Command errored out with exit status 1:
         command: 'C:\Users\luke1\Miniconda3\envs\m37\python.exe' -u -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'C:\\Users\\luke1\\AppData\\Local\\Temp\\pip-install-8238fwlv\\matplotlib_238ef439a6974f11be1e9134bb4d7c6f\\setup.py'"'"'; __file__='"'"'C:\\Users\\luke1\\AppData\\Local\\Temp\\pip-install-8238fwlv\\matplotlib_238ef439a6974f11be1e9134bb4d7c6f\\setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' install --record 'C:\Users\luke1\AppData\Local\Temp\pip-record-0b21nebk\install-record.txt' --single-version-externally-managed --compile --install-headers 'C:\Users\luke1\Miniconda3\envs\m37\Include\matplotlib'
             cwd: C:\Users\luke1\AppData\Local\Temp\pip-install-8238fwlv\matplotlib_238ef439a6974f11be1e9134bb4d7c6f\
        Complete output (499 lines):
        ================================================================================
        Edit setup.cfg to change the build options
        
        BUILDING MATPLOTLIB
          matplotlib: yes [3.1.0]
              python: yes [3.8.8 (default, Feb 24 2021, 15:54:32) [MSC v.1928 64 bit
                          (AMD64)]]
            platform: yes [win32]
        
        OPTIONAL SUBPACKAGES
         sample_data: yes [installing]
               tests: no  [skipping due to configuration]
        
        OPTIONAL BACKEND EXTENSIONS
                 agg: yes [installing]
               tkagg: yes [installing; run-time loading from Python Tcl/Tk]
              macosx: no  [Mac OS-X only]
        
        OPTIONAL PACKAGE DATA
                dlls: no  [skipping due to configuration]
        
        running install
        running build
        running build_py
        creating build
        creating build\lib.win-amd64-3.8
        copying lib\pylab.py -> build\lib.win-amd64-3.8
        creating build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\afm.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\animation.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\artist.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\axis.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\backend_bases.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\backend_managers.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\backend_tools.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\bezier.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\blocking_input.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\category.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\cm.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\collections.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\colorbar.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\colors.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\container.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\contour.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\dates.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\docstring.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\dviread.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\figure.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\fontconfig_pattern.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\font_manager.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\gridspec.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\hatch.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\image.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\legend.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\legend_handler.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\lines.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\markers.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\mathtext.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\mlab.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\offsetbox.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\patches.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\path.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\patheffects.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\pylab.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\pyplot.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\quiver.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\rcsetup.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\sankey.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\scale.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\spines.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\stackplot.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\streamplot.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\table.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\texmanager.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\text.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\textpath.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\ticker.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\tight_bbox.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\tight_layout.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\transforms.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\type1font.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\units.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\widgets.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\_animation_data.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\_cm.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\_cm_listed.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\_color_data.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\_constrained_layout.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\_layoutbox.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\_mathtext_data.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\_pylab_helpers.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\_version.py -> build\lib.win-amd64-3.8\matplotlib
        copying lib\matplotlib\__init__.py -> build\lib.win-amd64-3.8\matplotlib
        creating build\lib.win-amd64-3.8\mpl_toolkits
        copying lib\mpl_toolkits\__init__.py -> build\lib.win-amd64-3.8\mpl_toolkits
        creating build\lib.win-amd64-3.8\matplotlib\axes
        copying lib\matplotlib\axes\_axes.py -> build\lib.win-amd64-3.8\matplotlib\axes
        copying lib\matplotlib\axes\_base.py -> build\lib.win-amd64-3.8\matplotlib\axes
        copying lib\matplotlib\axes\_secondary_axes.py -> build\lib.win-amd64-3.8\matplotlib\axes
        copying lib\matplotlib\axes\_subplots.py -> build\lib.win-amd64-3.8\matplotlib\axes
        copying lib\matplotlib\axes\__init__.py -> build\lib.win-amd64-3.8\matplotlib\axes
        creating build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_agg.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_cairo.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_gtk3.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_gtk3agg.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_gtk3cairo.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_macosx.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_mixed.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_nbagg.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_pdf.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_pgf.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_ps.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_qt4.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_qt4agg.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_qt4cairo.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_qt5.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_qt5agg.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_qt5cairo.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_svg.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_template.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_tkagg.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_tkcairo.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_webagg.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_webagg_core.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_wx.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_wxagg.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\backend_wxcairo.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\qt_compat.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\tkagg.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\windowing.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\wx_compat.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\_backend_pdf_ps.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\_backend_tk.py -> build\lib.win-amd64-3.8\matplotlib\backends
        copying lib\matplotlib\backends\__init__.py -> build\lib.win-amd64-3.8\matplotlib\backends
        creating build\lib.win-amd64-3.8\matplotlib\cbook
        copying lib\matplotlib\cbook\deprecation.py -> build\lib.win-amd64-3.8\matplotlib\cbook
        copying lib\matplotlib\cbook\__init__.py -> build\lib.win-amd64-3.8\matplotlib\cbook
        creating build\lib.win-amd64-3.8\matplotlib\compat
        copying lib\matplotlib\compat\subprocess.py -> build\lib.win-amd64-3.8\matplotlib\compat
        copying lib\matplotlib\compat\__init__.py -> build\lib.win-amd64-3.8\matplotlib\compat
        creating build\lib.win-amd64-3.8\matplotlib\projections
        copying lib\matplotlib\projections\geo.py -> build\lib.win-amd64-3.8\matplotlib\projections
        copying lib\matplotlib\projections\polar.py -> build\lib.win-amd64-3.8\matplotlib\projections
        copying lib\matplotlib\projections\__init__.py -> build\lib.win-amd64-3.8\matplotlib\projections
        creating build\lib.win-amd64-3.8\matplotlib\sphinxext
        copying lib\matplotlib\sphinxext\mathmpl.py -> build\lib.win-amd64-3.8\matplotlib\sphinxext
        copying lib\matplotlib\sphinxext\plot_directive.py -> build\lib.win-amd64-3.8\matplotlib\sphinxext
        copying lib\matplotlib\sphinxext\__init__.py -> build\lib.win-amd64-3.8\matplotlib\sphinxext
        creating build\lib.win-amd64-3.8\matplotlib\style
        copying lib\matplotlib\style\core.py -> build\lib.win-amd64-3.8\matplotlib\style
        copying lib\matplotlib\style\__init__.py -> build\lib.win-amd64-3.8\matplotlib\style
        creating build\lib.win-amd64-3.8\matplotlib\testing
        copying lib\matplotlib\testing\compare.py -> build\lib.win-amd64-3.8\matplotlib\testing
        copying lib\matplotlib\testing\conftest.py -> build\lib.win-amd64-3.8\matplotlib\testing
        copying lib\matplotlib\testing\decorators.py -> build\lib.win-amd64-3.8\matplotlib\testing
        copying lib\matplotlib\testing\determinism.py -> build\lib.win-amd64-3.8\matplotlib\testing
        copying lib\matplotlib\testing\disable_internet.py -> build\lib.win-amd64-3.8\matplotlib\testing
        copying lib\matplotlib\testing\exceptions.py -> build\lib.win-amd64-3.8\matplotlib\testing
        copying lib\matplotlib\testing\__init__.py -> build\lib.win-amd64-3.8\matplotlib\testing
        creating build\lib.win-amd64-3.8\matplotlib\tri
        copying lib\matplotlib\tri\triangulation.py -> build\lib.win-amd64-3.8\matplotlib\tri
        copying lib\matplotlib\tri\tricontour.py -> build\lib.win-amd64-3.8\matplotlib\tri
        copying lib\matplotlib\tri\trifinder.py -> build\lib.win-amd64-3.8\matplotlib\tri
        copying lib\matplotlib\tri\triinterpolate.py -> build\lib.win-amd64-3.8\matplotlib\tri
        copying lib\matplotlib\tri\tripcolor.py -> build\lib.win-amd64-3.8\matplotlib\tri
        copying lib\matplotlib\tri\triplot.py -> build\lib.win-amd64-3.8\matplotlib\tri
        copying lib\matplotlib\tri\trirefine.py -> build\lib.win-amd64-3.8\matplotlib\tri
        copying lib\matplotlib\tri\tritools.py -> build\lib.win-amd64-3.8\matplotlib\tri
        copying lib\matplotlib\tri\__init__.py -> build\lib.win-amd64-3.8\matplotlib\tri
        creating build\lib.win-amd64-3.8\matplotlib\backends\qt_editor
        copying lib\matplotlib\backends\qt_editor\figureoptions.py -> build\lib.win-amd64-3.8\matplotlib\backends\qt_editor
        copying lib\matplotlib\backends\qt_editor\formlayout.py -> build\lib.win-amd64-3.8\matplotlib\backends\qt_editor
        copying lib\matplotlib\backends\qt_editor\formsubplottool.py -> build\lib.win-amd64-3.8\matplotlib\backends\qt_editor
        copying lib\matplotlib\backends\qt_editor\_formlayout.py -> build\lib.win-amd64-3.8\matplotlib\backends\qt_editor
        copying lib\matplotlib\backends\qt_editor\__init__.py -> build\lib.win-amd64-3.8\matplotlib\backends\qt_editor
        creating build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
        copying lib\matplotlib\testing\jpl_units\Duration.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
        copying lib\matplotlib\testing\jpl_units\Epoch.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
        copying lib\matplotlib\testing\jpl_units\EpochConverter.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
        copying lib\matplotlib\testing\jpl_units\StrConverter.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
        copying lib\matplotlib\testing\jpl_units\UnitDbl.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
        copying lib\matplotlib\testing\jpl_units\UnitDblConverter.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
        copying lib\matplotlib\testing\jpl_units\UnitDblFormatter.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
        copying lib\matplotlib\testing\jpl_units\__init__.py -> build\lib.win-amd64-3.8\matplotlib\testing\jpl_units
        creating build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\anchored_artists.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\angle_helper.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\axes_divider.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\axes_grid.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\axes_rgb.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\axes_size.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\axislines.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\axisline_style.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\axis_artist.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\clip_path.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\colorbar.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\floating_axes.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\grid_finder.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\grid_helper_curvelinear.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\inset_locator.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\parasite_axes.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        copying lib\mpl_toolkits\axes_grid\__init__.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid
        creating build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
        copying lib\mpl_toolkits\axes_grid1\anchored_artists.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
        copying lib\mpl_toolkits\axes_grid1\axes_divider.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
        copying lib\mpl_toolkits\axes_grid1\axes_grid.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
        copying lib\mpl_toolkits\axes_grid1\axes_rgb.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
        copying lib\mpl_toolkits\axes_grid1\axes_size.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
        copying lib\mpl_toolkits\axes_grid1\colorbar.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
        copying lib\mpl_toolkits\axes_grid1\inset_locator.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
        copying lib\mpl_toolkits\axes_grid1\mpl_axes.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
        copying lib\mpl_toolkits\axes_grid1\parasite_axes.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
        copying lib\mpl_toolkits\axes_grid1\__init__.py -> build\lib.win-amd64-3.8\mpl_toolkits\axes_grid1
        creating build\lib.win-amd64-3.8\mpl_toolkits\axisartist
        copying lib\mpl_toolkits\axisartist\angle_helper.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
        copying lib\mpl_toolkits\axisartist\axes_divider.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
        copying lib\mpl_toolkits\axisartist\axes_grid.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
        copying lib\mpl_toolkits\axisartist\axes_rgb.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
        copying lib\mpl_toolkits\axisartist\axislines.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
        copying lib\mpl_toolkits\axisartist\axisline_style.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
        copying lib\mpl_toolkits\axisartist\axis_artist.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
        copying lib\mpl_toolkits\axisartist\clip_path.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
        copying lib\mpl_toolkits\axisartist\floating_axes.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
        copying lib\mpl_toolkits\axisartist\grid_finder.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
        copying lib\mpl_toolkits\axisartist\grid_helper_curvelinear.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
        copying lib\mpl_toolkits\axisartist\parasite_axes.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
        copying lib\mpl_toolkits\axisartist\__init__.py -> build\lib.win-amd64-3.8\mpl_toolkits\axisartist
        creating build\lib.win-amd64-3.8\mpl_toolkits\mplot3d
        copying lib\mpl_toolkits\mplot3d\art3d.py -> build\lib.win-amd64-3.8\mpl_toolkits\mplot3d
        copying lib\mpl_toolkits\mplot3d\axes3d.py -> build\lib.win-amd64-3.8\mpl_toolkits\mplot3d
        copying lib\mpl_toolkits\mplot3d\axis3d.py -> build\lib.win-amd64-3.8\mpl_toolkits\mplot3d
        copying lib\mpl_toolkits\mplot3d\proj3d.py -> build\lib.win-amd64-3.8\mpl_toolkits\mplot3d
        copying lib\mpl_toolkits\mplot3d\__init__.py -> build\lib.win-amd64-3.8\mpl_toolkits\mplot3d
        creating build\lib.win-amd64-3.8\matplotlib\mpl-data
        creating build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\back.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        creating build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts
        creating build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Courier-Bold.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        creating build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\afm\cmex10.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        creating build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\stylelib\seaborn-deep.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\images\subplots_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        creating build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSerif.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend
        creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
        creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\images
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\images\ui-icons_ffffff_256x240.png -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\images
        copying lib\matplotlib\mpl-data\fonts\afm\phvlo8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Times-Bold.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        copying lib\matplotlib\mpl-data\fonts\ttf\cmtt10.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Times-BoldItalic.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        copying lib\matplotlib\mpl-data\stylelib\seaborn-talk.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\stylelib\grayscale.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\images\move_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Times-Italic.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        creating build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\sample_data\percent_bachelors_degrees_women_usa.csv -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\stylelib\seaborn-notebook.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\fonts\afm\pcrro8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\sample_data\eeg.dat -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\external
        creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\external\jquery
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\external\jquery\jquery.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\external\jquery
        creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\css
        copying lib\matplotlib\backends\web_backend\css\boilerplate.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\css
        copying lib\matplotlib\mpl-data\images\hand.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\sample_data\None_vs_nearest-pdf.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\sample_data\grace_hopper.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\images\filesave.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\stylelib\seaborn-dark-palette.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\sample_data\membrane.dat -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\AUTHORS.txt -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
        copying lib\matplotlib\mpl-data\fonts\afm\phvro8an.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\images\filesave.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\afm\ptmri8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\images\filesave_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\stylelib\classic.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\fonts\afm\ptmbi8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\stylelib\seaborn-darkgrid.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\sample_data\demodata.csv -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\sample_data\goog.npz -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\js
        copying lib\matplotlib\backends\web_backend\js\mpl.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\js
        copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Helvetica-BoldOblique.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        copying lib\matplotlib\mpl-data\sample_data\s1045.ima.gz -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXGeneralBolIta.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\images\back_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\images\ui-icons_555555_256x240.png -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\images
        copying lib\matplotlib\mpl-data\fonts\ttf\cmss10.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\images\zoom_to_rect.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\ttf\cmmi10.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\images\matplotlib.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSans-BoldOblique.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\images\hand.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizOneSymReg.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\backends\web_backend\css\fbm.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\css
        copying lib\matplotlib\mpl-data\fonts\afm\phvl8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\images\back.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\stylelib\seaborn-colorblind.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\images\home.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\move.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSansMono-Oblique.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\images\move.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\home_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Courier-BoldOblique.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        copying lib\matplotlib\mpl-data\stylelib\ggplot.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Helvetica.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        copying lib\matplotlib\mpl-data\fonts\afm\putr8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery
        creating build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery\js
        copying lib\matplotlib\backends\web_backend\jquery\js\jquery.min.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery\js
        copying lib\matplotlib\mpl-data\fonts\afm\pbkl8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\images\qt4_editor_options.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\images\ui-icons_cc0000_256x240.png -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\images
        copying lib\matplotlib\mpl-data\images\zoom_to_rect_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\stylelib\seaborn-ticks.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\fonts\ttf\cmb10.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXNonUniIta.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\afm\pplb8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\images\forward.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\afm\pncbi8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXNonUniBolIta.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\images\filesave.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\help.ppm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\back.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\help.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\stylelib\seaborn-bright.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\images\move_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\subplots.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\matplotlib.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\filesave.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\stylelib\seaborn-pastel.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\fonts\afm\pcrr8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\images\home_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizFourSymReg.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\afm\pbkd8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSansMono-Bold.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\images\forward.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\matplotlib.ppm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\backends\web_backend\single_figure.html -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend
        copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\readme.txt -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.theme.min.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
        copying lib\matplotlib\mpl-data\fonts\afm\ptmr8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\sample_data\README.txt -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSans-Bold.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\afm\phvro8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXGeneralItalic.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXNonUniBol.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\images\subplots.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\afm\phvbo8an.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizFiveSymReg.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\images\subplots.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\sample_data\embedding_in_wx3.xrc -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\stylelib\fivethirtyeight.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\sample_data\Minduka_Present_Blue_Pack.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizFourSymBol.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSans-Oblique.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\backends\web_backend\jquery\js\jquery.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery\js
        copying lib\matplotlib\mpl-data\fonts\afm\pagdo8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\sample_data\ada.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\stylelib\seaborn-whitegrid.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\fonts\afm\pbkdi8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\afm\phvbo8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXNonUni.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\images\subplots_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\afm\pplr8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\images\move.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\backends\web_backend\js\nbagg_mpl.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\js
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizOneSymBol.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\ttf\cmsy10.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\images\zoom_to_rect.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\forward.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\help.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\help_large.ppm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
        copying lib\matplotlib\mpl-data\fonts\afm\pagko8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\index.html -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
        copying lib\matplotlib\mpl-data\images\move.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\ttf\LICENSE_DEJAVU -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\stylelib\seaborn-white.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\images\home.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\afm\pcrb8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\images\ui-icons_777620_256x240.png -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\images
        copying lib\matplotlib\mpl-data\fonts\afm\pagk8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\images\forward.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\afm\pplbi8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\stylelib\Solarize_Light2.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\fonts\afm\cmr10.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\images\forward_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Courier-Oblique.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        copying lib\matplotlib\mpl-data\sample_data\jacksboro_fault_dem.npz -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\images\matplotlib.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\qt4_editor_options_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\afm\pncb8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\stylelib\seaborn-dark.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\stylelib\dark_background.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\stylelib\seaborn.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\images\back_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.theme.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\images\ui-icons_444444_256x240.png -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\images
        copying lib\matplotlib\mpl-data\fonts\afm\cmmi10.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\afm\phvr8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\matplotlibrc -> build\lib.win-amd64-3.8\matplotlib\mpl-data
        copying lib\matplotlib\mpl-data\fonts\ttf\cmr10.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\stylelib\seaborn-muted.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\ZapfDingbats.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        copying lib\matplotlib\mpl-data\sample_data\grace_hopper.jpg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizTwoSymBol.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\images\matplotlib_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Times-Roman.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.min.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
        copying lib\matplotlib\backends\web_backend\ipython_inline_figure.html -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend
        copying lib\matplotlib\mpl-data\images\subplots.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\LICENSE.txt -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
        copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSansDisplay.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizThreeSymBol.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\images\hand.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSerifDisplay.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\afm\psyr.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\afm\pzcmi8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\ttf\LICENSE_STIX -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSans.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSerif-BoldItalic.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\sample_data\msft.csv -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\stylelib\_classic_test.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.structure.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
        copying lib\matplotlib\mpl-data\sample_data\data_x_x2_x3.csv -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\images\help.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\sample_data\logo2.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        creating build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data\axes_grid
        copying lib\matplotlib\mpl-data\sample_data\axes_grid\bivariate_normal.npy -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data\axes_grid
        copying lib\matplotlib\mpl-data\stylelib\fast.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\stylelib\seaborn-paper.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizThreeSymReg.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\afm\pzdr.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\images\help_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\zoom_to_rect.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\hand.pdf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\home.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\stylelib\bmh.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\fonts\afm\pcrbo8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\package.json -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
        copying lib\matplotlib\mpl-data\sample_data\topobathy.npz -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\images\zoom_to_rect.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\stylelib\tableau-colorblind10.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\fonts\afm\ptmb8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\images\qt4_editor_options.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\afm\pagd8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\backends\web_backend\nbagg_uat.ipynb -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\images\ui-icons_777777_256x240.png -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1\images
        copying lib\matplotlib\mpl-data\fonts\ttf\cmex10.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\afm\putbi8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.min.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
        copying lib\matplotlib\mpl-data\images\zoom_to_rect_large.png -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\backends\web_backend\jquery-ui-1.12.1\jquery-ui.structure.min.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\jquery-ui-1.12.1
        copying lib\matplotlib\mpl-data\images\qt4_editor_options.svg -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\images\home.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\stylelib\seaborn-poster.mplstyle -> build\lib.win-amd64-3.8\matplotlib\mpl-data\stylelib
        copying lib\matplotlib\mpl-data\fonts\afm\phvb8an.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSerif-Italic.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\images\filesave_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\mpl-data\fonts\afm\cmsy10.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\afm\putri8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXGeneralBol.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\sample_data\ct.raw.gz -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        copying lib\matplotlib\mpl-data\fonts\afm\putb8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSerif-Bold.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Helvetica-Bold.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXGeneral.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\images\back.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\backends\web_backend\css\page.css -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\css
        copying lib\matplotlib\mpl-data\images\hand_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\backends\web_backend\js\mpl_tornado.js -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend\js
        copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Helvetica-Oblique.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        copying lib\matplotlib\mpl-data\fonts\ttf\STIXSizTwoSymReg.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Courier.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        copying lib\matplotlib\mpl-data\fonts\afm\phvb8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\afm\pplri8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\images\forward_large.gif -> build\lib.win-amd64-3.8\matplotlib\mpl-data\images
        copying lib\matplotlib\backends\web_backend\all_figures.html -> build\lib.win-amd64-3.8\matplotlib\backends\web_backend
        copying lib\matplotlib\mpl-data\fonts\pdfcorefonts\Symbol.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\pdfcorefonts
        copying lib\matplotlib\mpl-data\fonts\afm\pbkli8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\afm\pncr8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\afm\phvr8an.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\afm\pncri8a.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSansMono.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\ttf\DejaVuSansMono-BoldOblique.ttf -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\ttf
        copying lib\matplotlib\mpl-data\fonts\afm\cmtt10.afm -> build\lib.win-amd64-3.8\matplotlib\mpl-data\fonts\afm
        copying lib\matplotlib\mpl-data\sample_data\aapl.npz -> build\lib.win-amd64-3.8\matplotlib\mpl-data\sample_data
        UPDATING build\lib.win-amd64-3.8\matplotlib\_version.py
        set build\lib.win-amd64-3.8\matplotlib\_version.py to '3.1.0'
        running build_ext
        building 'matplotlib.ft2font' extension
        error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
        ----------------------------------------
    ERROR: Command errored out with exit status 1: 'C:\Users\luke1\Miniconda3\envs\m37\python.exe' -u -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'C:\\Users\\luke1\\AppData\\Local\\Temp\\pip-install-8238fwlv\\matplotlib_238ef439a6974f11be1e9134bb4d7c6f\\setup.py'"'"'; __file__='"'"'C:\\Users\\luke1\\AppData\\Local\\Temp\\pip-install-8238fwlv\\matplotlib_238ef439a6974f11be1e9134bb4d7c6f\\setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' install --record 'C:\Users\luke1\AppData\Local\Temp\pip-record-0b21nebk\install-record.txt' --single-version-externally-managed --compile --install-headers 'C:\Users\luke1\Miniconda3\envs\m37\Include\matplotlib' Check the logs for full command output.
    


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-1-0800079e47ce> in <module>
          3 get_ipython().system('pip install -r requirements.txt')
          4 
    ----> 5 import ipywidgets as widgets
          6 from ipywidgets import interact, interactive, fixed, interact_manual
          7 from IPython.display import clear_output, Image
    

    ModuleNotFoundError: No module named 'ipywidgets'


# The Bitter Lesson

Rich Sutton, one of the pioneers of modern reinforcement learning, detailed an account of the ["Bitter Lesson"](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) of machine learning. 
               
In short, he speaks of the power of computation-driven learning. Over the almost 100 years of serious AI research that has happened so much effort has been put into distilling human knowledge down to core concepts or primitives that we can give to machines to make them intelligent. Time and time again, the biggest breakthoughs in ML/AI have not been in perfect representations of knowledge or heuristics about how to learn, but in leveraging computation resources available to allow these software agents to discover the structure in the problems themselves.

## Learning without Knowledge

The purpose of this session is to show some techiques we have under our belt to tackles problems without incorporating knowledge of the process we want to learn. While using heuristics or priors can great speed up learning on certain problems, we want to show in general we can learn in the most efficient way possible without any of that.

We will structure this post into 3 parts:

* Stateless, only your actions yield outcomes across independent trails
* Stateful, your actions yield outcomes in a context across independent trials
* Stateful, your actions yield outcomes in a context over time

This allows us to tackle problems with minimum information about the world first, see how we would solve them, and then modify our policies when we have more information to take advantage.


## Part 1: Bandits



![Bandit](assets/multi_armed_bandit.png)

A bandit problem refers to an environment where only the agents actions will effect the reward they recieve. The example above shows that each action (the arm to pull) has a certain distribution of rewards that can result. The job of the agent is to find out as quickly as possible which action is the best. 

What does "best" mean in this context? The metric that is used is called the **Regret**.

### Regret

It is defined as the expected reward an agent lost out on by not choosing the optimal action. To put a bit of terminology to this, we can define the expected reward of an action as $Q(a_i) = E[r | a]$, also called the **Value Function**. If we assume that the bandit arms will give rewards according to a bernoulli random variable $r(a_i) \sim bern(\theta_i)$. Then the regret can be written as:

$$
R(t) = \sum_{t = 0}^T (\theta_* - Q(a_t))
$$



For example, consider a 2-armed bandit with pull probabilities $(0.3, 0.8)$. The optimal action is to choose the second arm always, $\theta_* = 0.8$. We can compare to action sequences using the regret. Imagine action sequence 1:  $[0, 1, 1, 1, 0, 0]$ and action sequence 2: $[0, 0, 0, 0, 1, 1]$. The regret they accrue can be computed with the above formula:



$$
R_1(t) = (0.8 - 0.3) + (0.8 - 0.8) + (0.8 - 0.8) + (0.8 - 0.8) + (0.8 - 0.3) + (0.8 - 0.3) = 1.5 \\
R_2(t) = (0.8 - 0.3) + (0.8 - 0.3) + (0.8 - 0.3) + (0.8 - 0.3) + (0.8 - 0.8) + (0.8 - 0.8) = 2
$$

So the policy that generated the first sequence has the least regret. We seek to find policies with the smallest expected regret.


```python
"""
Let's play around with a bandit to see what the problem is like. You can change any of the variables below
to modify how many arms there are or the likelihoods of arms giving reward. Once you run this cell, you will be 
able to interact with some buttons to produce plots on how well you are doing.
"""
n_arms = 8

# Setup some random arm probabilities
bad_arm_prob = 0.3
good_arm_prob = 0.8
regret = good_arm_prob - bad_arm_prob
probs = np.zeros(n_arms) + bad_arm_prob
probs[np.random.choice(range(n_arms))] = good_arm_prob

# Configure environment
config = {"num_arms" : n_arms, "probs": probs}
env = Bandit(config)

# Create fancy UI
buttons = [widgets.Button(description=f'Arm: {i}') for i in range(n_arms)]
cumulative_reward = [0]
def press(button):
    ## Sanitize inputs, clear current output
    clear_output()
    value = int(button.description.split(" ")[1])
    
    ## Step env, keep track of cumulative reward each step
    r = env.step(value)
    cumulative_reward.append(env.cumulative_reward.sum())
    print(f"Reward: {r}, Total steps: {env.visits.sum()}")
    
    ## plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 8))
    plot_avg_reward(axs[0], env)
    plot_cumulative_reward(axs[1], cumulative_reward)
    plot_cumulative_regret(axs[2], regret, env)

    plt.legend()
    fig.tight_layout()
    
    display(buttons)

[buttons[i].on_click(press) for i in range(n_arms)]
buttons = widgets.HBox(buttons)
display(buttons)
```

    Reward: 1, Total steps: 17.0
    


    HBox(children=(Button(description='Arm: 0', style=ButtonStyle()), Button(description='Arm: 1', style=ButtonSty…



    
![png](tree_search_files/tree_search_7_2.png)
    


## Solving bandit problems

As we can see in the regret formula, the most important parameter to estimate is the true average reward an arm will yield.  
Without any model of how the reward-generating process works, the agent needs to collect information and construct an estimate of the arm averages. 

**Exploration** and **Exploitation** describes the tradeoff an agent makes while it is trying to make these decisions online. The strategies we will investigate different combinations of how to make decisions under uncertainty:
- random exploration
- minimal exploration, just greedily pull best values we have seen in past
- explore with preference towards uncertainty


```python
"""
Pure random exploration will investigate each action with equal likelihood, but never exploits that knowledge to improve its regret.
"""

# Run 500 trials of 100 step bandit interactions
n_steps = 100
n_trials = 500
n_arms = 8

# Arm probabilities 
config = {"num_arms" : n_arms, "probs": (1/n_arms) * np.arange(n_arms)}
env = Bandit(config)

def random_selection(env):
    return np.random.randint(env.n_arms)

random = run_simulation(random_selection, env, n_steps=n_steps, n_trials=n_trials)
plot_simulation([random[1]], random[0], env)
plt.show()
```


    
![png](tree_search_files/tree_search_9_0.png)
    



```python
"""
The epsilon greedy action selection algorithm uses the same ideas from random exploration, 
but ensures we exploit that information some fraction of the time. 

Several variations of this algorithm can be constructed, such as using an annealing schedule to drop the
epsilon value over time when we believe our arm average estimates are trustworthy to more greedily apply.
"""
def e_greedy_selection(env, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(env.n_arms)
    else:
        return np.argmax(env.arm_avgs())
def create_egreedy(epsilon):
    "Helper method to return a handle for different epsilon values"
    return lambda env: e_greedy_selection(env, epsilon)
def update(epsilon):
    global egreedy
    selection = create_egreedy(epsilon)
    egreedy = run_simulation(selection, env, n_steps=n_steps, n_trials=n_trials)
    plot_simulation([random[1], egreedy[1]], egreedy[0], env)
    plt.show()
# Allows you to interact with different epsilon values and see their effect on solutions relative to the previous random policy
interact_manual(update, epsilon=widgets.FloatSlider(min=0, max=1, step=0.05))
```


    interactive(children=(FloatSlider(value=0.0, description='epsilon', max=1.0, step=0.05), Button(description='R…





    <function __main__.update(epsilon)>



### UCB Algorithm

$\epsilon$-Greedy works well for a lot of problems, but requires tuning of the hyperparameter is not really principled in any sort of way. This makes it difficult to say anything about the theoretical guarantees of the algorithm. Using some simple arguments about probability bounds, we can find better tradeoffs of exploitation and exploration.

The idea is to try and estimate how likely a big gap is between our estimated value function and the true one: $Q(a) - \hat{Q}_t(a) \leq \hat{U}_t(a)$. Hoeffding's inequality can be used if we do not want to assign any prior to the distribution of rewards, works for any bounded distribution, so for a random variable $r$ bounded between $[c_1, c_2]$:

$$P[\frac{1}{n} \sum_{i =0}^{n}(r_i - E[r_i]) \geq \Delta\ ] \leq e^{- \frac{2n\Delta^2}{(c_2 - c_1)^2}}$$

This looks eerily similar to our estimates of our value function, $Q(a) = \frac{1}{n} \sum_{i=0}^{n} r_i$:


$$P[Q_t - \hat{Q_t} \geq \Delta ] \leq e^{-\frac{2n(a)\Delta^2}{(c_2 - c_1)^2}}$$

## UCB 1 Heuristic

We see that then our upper bound on the true reward can be given by $\Delta = \hat{U_t}(a)$. And if we make the simplifying assumption that the reward is bounded to be on the interval $(0, 1)$. We can then for any value of the probability of deviation, $p$, we can solve for what we expect the upper bound on that deviation to be:

$$
p \approx e^{-2n(a)U_t(a)^2} \\
U_t(a) = \sqrt{\frac{- \log p}{2 n (a)}}
$$

Most of the time in practice, we use a heuristic to drop this $p$ over time, typically $p=t^{-4}$. This yields the most commonly used version of UCB, UCB-1:

$$
U_t(a) = \sqrt{\frac{2 \log t}{n(a)}} \\
a_t = argmax_a [Q(a) + U_t(a)]
$$


```python
"""
UCB-1 action selection gives us a heuristic that works relatively well as long as the arm probabilities are not too close.
"""

def ucb_selection(env):
    ucb = np.sqrt((2 * np.log(env.visits.sum())) / env.visits)
    mod_avgs = env.arm_avgs() + ucb
    return np.argmax(mod_avgs)

ucb = run_simulation(ucb_selection, env, n_steps=n_steps, n_trials=n_trials)
plot_simulation([random[1], egreedy[1], ucb[1]], ucb[0], env)
plt.show()
```


    
![png](tree_search_files/tree_search_15_0.png)
    


## Part 2: Contextual Bandits

The next class of environments we will look at are referred to as "Contexual" bandit problems. This refers to an observation we get to perform before choosing our action. The observation allows us to have "context" of the situation to use in choosing a useful action.

![](assets/3_contextual_bandits.max-1500x1500.png)

Examples of these types of problems are very common in recommender systems. We can think of a user having a context (or in a profile) that we can review before making a decision as to what ad we should serve or movie we can recommend.


```python
"""
Let's play around with a bandit to see what the problem is like. You can change any of the variables below
to modify how many arms there are or the likelihoods of arms giving reward. Once you run this cell, you will be 
able to interact with some buttons to produce plots on how well you are doing.
"""
def draw_state(state):
        # Use a colored box to show context
    if (state == 0):
        return widgets.HTML(value="<div style='background-color: green ; padding: 10px; border: 1px solid green;'>")
    else: 
        return widgets.HTML(value="<div style='background-color: red ; padding: 10px; border: 1px solid red;'>")

n_arms = 8

n_arms = 2
n_states = 2

## Red: arm 1, Green: arm 0
state_probs = np.array([[0.1, 0.8], [0.8, 0.1]])
# state_probs = get_state_probs(n_states, n_arms)
config = {
    "num_arms" : n_arms,
    "num_states" : n_states,
    "state_probs": state_probs
}
env = ContextualBandit(config)
state = env.reset()
print(f"State: {state}")
cumulative_reward = [0]

# Create fancy UI
buttons = [widgets.Button(description=f'Arm: {i}') for i in range(n_arms)]
box = draw_state(state)
def press(button):
    global state, cumulative_reward
    
    ## Sanitize inputs, clear current output
    clear_output()
    value = int(button.description.split(" ")[1])
    
    ## Step env, keep track of cumulative reward each step
    state, reward = env.step(value)
    cumulative_reward.append(env.cumulative_reward.sum())
    print(f"New State: {state}, Reward: {reward}, Total steps: {env.visits.sum()}")

    
    fig, axs = plt.subplots(1, 3, figsize=(15, 8))

    
    plot_avg_reward_contextual(axs[0], env)
    plot_cumulative_reward(axs[1], cumulative_reward)
    plot_cumulative_regret(axs[2], 0.7, env)
    
    box = draw_state(state)
    fig.tight_layout()
    display(buttons)
    display(box)

[buttons[i].on_click(press) for i in range(n_arms)]
buttons = widgets.HBox(buttons)
display(buttons)
display(box)
```

    New State: 0, Reward: 1, Total steps: 13.0
    


    HBox(children=(Button(description='Arm: 0', style=ButtonStyle()), Button(description='Arm: 1', style=ButtonSty…



    HTML(value="<div style='background-color: green ; padding: 10px; border: 1px solid green;'>")



    
![png](tree_search_files/tree_search_17_3.png)
    



```python
"""
Similar as before, using no exploitation gives good coverage across the state/action space.

Note: the heat map on the left is normalized by row to show what the pulls were across all states.
"""
n_steps = 100
n_trials = 500
n_arms = 5
n_states = 10

n_steps *= n_states # more states dilutes each interaction
state_probs = get_state_probs(n_states, n_arms)
config = {
    "num_arms" : n_arms,
    "num_states" : n_states,
    "state_probs": state_probs
}
env = ContextualBandit(config)

def random_selection(env):
    return np.random.randint(env.n_arms)

random = run_simulation(random_selection, env, n_steps=n_steps, n_trials=n_trials)
plot_contextual_simulation([random[1]], random[0], env)
plt.show()
```


    
![png](tree_search_files/tree_search_18_0.png)
    



```python
"""
Similar to epsilon greedy as before, we keep track of the average rewards for each state for all the arms. 
We can then exploit this knowledge every so often dependent on the value of epsilon.
"""
def e_greedy_selection(env, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(env.n_arms)
    else:
        return np.argmax(env.arm_avgs(env.state))

def create_egreedy(epsilon):
    return lambda env: e_greedy_selection(env, epsilon)

def update(epsilon):
    global egreedy
    selection = create_egreedy(epsilon)
    egreedy = run_simulation(selection, env, n_steps=n_steps, n_trials=n_trials)
    plot_contextual_simulation([random[1], egreedy[1]], egreedy[0], env)
    plt.show()

interact_manual(update, epsilon=widgets.FloatSlider(min=0, max=1, step=0.05))
```


    interactive(children=(FloatSlider(value=0.0, description='epsilon', max=1.0, step=0.05), Button(description='R…





    <function __main__.update(epsilon)>




```python
"""
As with the other solutions we can simply construct similar estimates as the standard bandit problem, but now stateful.
"""
def ucb_selection(env):
    ucb = np.sqrt((2 * np.log(env.visits[env.state, :].sum())) / env.visits[env.state, :])
    mod_avgs = env.arm_avgs(env.state) + ucb
    return np.argmax(mod_avgs)

ucb = run_simulation(ucb_selection, env, n_steps=n_steps, n_trials=n_trials)
plot_contextual_simulation([random[1], egreedy[1], ucb[1]], ucb[0], env)
plt.show()
```


    
![png](tree_search_files/tree_search_20_0.png)
    


## Part 3: Markov Decision Processes (MDPs)

We have been building up the problem description in a more and more complex manner over the course of this session.

Our agents have only seen a state, took an action, and then recieved a reward for how they acted. There was no notion of time in any of these algorithms and so the agents actions had no effect on the future. Real decision making processes have some degree of effect on the payoff you get. We can model these systems with MDPs.

MDPs are a general framework for dealing with decision making over some sort of horizon (possibly even infinite).



For these next set of experiments, its helps intuition to think about making decisions in a spatial manner. 


```python
env = FrozenLakeEnv(8)
state = env.reset()
env.render()
```


    
![png](tree_search_files/tree_search_23_0.png)
    


## Value Functions

Just as with the bandit problems, a value function helped us decide which actions were high quality and those that were not. The only difference between value functions in MDPs and those in Bandits is that added consideration of time. To formalize a bit: 

$$
Q_t(s, a) = E [ \sum_{k = 0}^{\infty} R_{t + k + 1} | S_t = s, A_t = a]   
$$

This says the value of a state and an action is what expect the rest of the rewards to be for the rest of time. This is a powerful idea, because it allows us to turn each step of decision making back into a bandit problem. 

Let's take a look at a value function for the frozen lake environment.


```python
"""
The value function can be computed using dynamic programming. If we start from the goal state, we can iterate backwards until we reach the goal.
In order to encourage an agent to get to the goal as soon as possible, we discount the reward a state gets the longer it takes to get there.
"""

value_fcn = value_iteration(env, gamma=0.9)
env.render()
plot_value_function(value_fcn, env)
```


    
![png](tree_search_files/tree_search_25_0.png)
    



    
![png](tree_search_files/tree_search_25_1.png)
    


# Monte Carlo Tree Search

We have discussed MCTS on this team quite a bit, but how does it fit into all of this?
![](assets/EieiQ.png)

The algorithm essentially builds up this value function using a combination of tree search + random play (monte carlo simulation)


```python
# mcts = MCTS(env)
# env.get_action(mcts.next_move(state))

# print("Value computed during tree search: ")
# plot_value_function(mcts.value_fcn(), env)

print("Visitations for each state during tree search: ")
plot_value_function(mcts.visitation_fcn(), env)
```

    Visitations for each state during tree search: 
    


    
![png](tree_search_files/tree_search_27_1.png)
    



```python
# state = env.reset()
done = False

# env = FrozenLakeEnv(10)
state = env.reset()
env.render()
steps = 0
while not done:
    mcts = MCTS(env)
    action = mcts.next_move(state)
    state, reward, done, _ = env.step(action)
    steps += 1

    env.render(save=True, ind=steps)
    # plot_value_function(mcts.value_fcn(), env)
    
create_gif("assets/frozen_lake/step_", steps)

print(f"Success! {steps} moves")

clear_output(wait=True)
plt.clf()
Image(url='assets/solution.gif')  
```




<img src="assets/solution.gif"/>




    <Figure size 432x288 with 0 Axes>



```python

```
