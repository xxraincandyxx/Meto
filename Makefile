SWIFT_SOURCES := $(wildcard Sources/*.swift lessons/*/*.swift)
.build/metal-lab: $(SWIFT_SOURCES) Makefile
	@mkdir -p .build
	xcrun --sdk macosx swiftc -O -g -module-cache-path .build/modules $(SWIFT_SOURCES) -o $@
