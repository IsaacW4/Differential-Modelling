# Flooding Small Villages
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)
](https://www.python.org/dev/peps/pep-0008/)

Shared codebase for "MATH6149 CW2, Electric Boogaloo: Flooding".
Remember to create new branches for new features and please avoid 
making changes directly to `master`.

Have given some guidance below for setting up a local environment to
work with the code in this repository, it's by no means exhaustive but 
will hopefully save you the painful googling that I've often found comes
with this territory.

### Getting Set:
1) Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html
   ) (https://docs.conda.io/en/latest/miniconda.html) Python 3.8. When 
   installing, you'll probably not want to add Miniconda to PATH or
   make it your default Python installation, so avoid selecting these 
   options during the installation process.
2) Download and set up [PyCharm](https://www.jetbrains.com/pycharm
   ) (https://www.jetbrains.com/pycharm), please read note on this, 
   (it's a pretty intuitive and powerful IDE that plays nicely with
   this sort of stuff, but if you have preference for something like
   VSCode feel free to use that instead). Pycharm has a free version
   and a paid version, but [Student Licences](
   https://www.jetbrains.com/shop/eform/students) 
   (https://www.jetbrains.com/shop/eform/students) can be obtained
   here for free, just requires a university email. You may also find
   installing and maintaining your copy of PyCharm easier with the
   [JetBrains Toolbox App](https://www.jetbrains.com/toolbox-app/
   ) (https://www.jetbrains.com/toolbox-app/).
3) Create a new PyCharm project from this repository:
   1) Create project, call it `Flooding Villages` or something equally
    fun, project creation dialogue might look something like
    [this](https://f.coldcurry.net/HKm2
    ) (https://f.coldcurry.net/HKm2)
   2) Click (in the top menu bar) `VCS`, `Enable Version Control`.
   3) Click [Git, Manage Remotes](https://f.coldcurry.net/lbxW)
   4) Add a remote with this URL:
   `https://git.soton.ac.uk/mde-kiiking-group/flooding-small-villages.git`,
      do not drop the `.git` at the end.
   5) Click [Git, Fetch](https://f.coldcurry.net/B9ZY)
   6) `Checkout` the `origin/master` remote branch ([gif here](
      https://f.coldcurry.net/eXVx.gif)).
4) You've now got the latest copy of the shared project code saved locally,
the final remaining step is to download and install project dependencies
   (which is managed automatically by Conda):
   1) Open a new terminal instance inside PyCharm
   2) Execute the command `conda env update --file environment.yml`
      ([example](https://f.coldcurry.net/vRyc.gif))
   3) Once the terminal has finished working ([example](
      https://f.coldcurry.net/GCRV)) you're good to go.
5) Everything should now be ready to go, try running 
   `flooding.sol.godunov` to confirm that you get a sensible plot.

### Formatting:
Since it's likely a lot of this codebase is going to be shared, some
level of consistence in formatting is likely a good idea to establish.
Wherever possible, aim to stick to PEP 8's guidance for Python 
formatting. In some circumstances, it is reasonable to exceed the `79`
character line limit, though docstrings and comments should remain
at or under their limit of `72` characters. The [file](./pep8_style.xml) 
`./pep8_style.xml`may be imported in PyCharm to add line limit markers
at the locations`72`, `79`, and `100`, with a "hard" limit at `79`
characters (the IDE will warn you when lines exceed this length. 
You can import this file by navigating to the 
[following location in your settings](https://f.coldcurry.net/pU5l.png) 
and importing the file.
