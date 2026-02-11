./: {*/ -build/ -certifiable-build/ -include/ -.github/ -docs/} \
  doc{README.md} \
  legal{LICENSE} \
  manifest

./: src/ (examples/) tests/

import src = src/
import examples = examples/
import tests = tests/

# Don't install tests.
#
tests/: install = false
