.. image:: https://travis-ci.org/molmod/yaff.svg?branch=master
    :target: https://travis-ci.org/molmod/yaff
.. image:: https://ci.appveyor.com/api/projects/status/bwgl6x2wslkv09so/branch/master?svg=true
    :target: https://ci.appveyor.com/project/molmod-bot/yaff
.. image:: https://anaconda.org/molmod/yaff/badges/version.svg
    :target: https://anaconda.org/molmod/yaff
.. image:: https://codecov.io/gh/molmod/yaff/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/molmod/yaff

Yaff stands for "Yet another force field". It is a pythonic force-field code
used by Toon and Louis to test-drive their new models. The original motivation
to develop Yaff was to provide a good reference implementation of the force
fields developed at the Center for Molecular Modeling (CMM) at Ghent University.
In its current version, Yaff is general and flexible enough to handle a large
variety of force field models.

More information about Yaff can be found on the CMM Code website:
http://molmod.ugent.be/software

Yaff is distributed as open source software under the conditions of the GPL
license version 3. Read the file COPYING for more details, or visit
http://www.gnu.org/licenses/

**FORKED**


Installation
============

Yaff can be installed with pip (system wide or in a virtual environment):

.. code:: bash

    pip install numpy Cython
    pip install yaff

Alternatively, you can install Yaff in your home directory:

.. code:: bash

    pip install numpy Cython --user
    pip install yaff --user

Lastly, you can also install MolMod with conda. (See
https://www.continuum.io/downloads)

.. code:: bash

    conda install -c molmod yaff

Testing
=======

The tests can be executed as follows:

.. code:: bash

    nosetests yaff
