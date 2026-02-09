./: {*/ -build/} doc{README.md PACKAGE-README.md} legal{LICENSE} manifest

# Don't install tests.
#
tests/: install = false
