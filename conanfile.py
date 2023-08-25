from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps


class kvexe_rocksdbRecipe(ConanFile):
    name = "kvexe-rocksdb"
    version = "0.1.0"
    package_type = "application"

    # Optional metadata
    license = "GPLv3"
    author = "Jiansheng Qiu jianshengqiu.cs@gmail.com"

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*"

    options = {
        "ROCKSDB_INCLUDE": ["ANY"],
        "ROCKSDB_LIB": ["ANY"],
    }

    def requirements(self):
        self.requires("counter-timer/[>=0.1.1]")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cli_args=[
            "-DROCKSDB_INCLUDE=" + str(self.options.ROCKSDB_INCLUDE),
            "-DROCKSDB_LIB=" + str(self.options.ROCKSDB_LIB),
        ]
        cmake.configure(cli_args=cli_args)
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
