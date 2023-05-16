from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps


class kvexe_kvexeRecipe(ConanFile):
    name = "kvexe-kvexe"
    version = "0.1.0"
    package_type = "application"

    # Optional metadata
    license = "GPLv3"
    author = "Jiansheng Qiu jianshengqiu.cs@gmail.com"
    topics = ("<Put some tag here>", "<here>", "<and here>")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*", "3rdparty/*"

    options = {
        "ROCKSDB_INCLUDE": ["ANY"],
        "ROCKSDB_LIB": ["ANY"],
        "VISCNTS_INCLUDE": ["ANY"],
        "VISCNTS_LIB": ["ANY"],
    }

    def requirements(self):
        self.requires("rcu-vector/[~0.1]")

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
            "-DVISCNTS_INCLUDE=" + str(self.options.VISCNTS_INCLUDE),
        ]
        viscnts_lib = self.options.get_safe("VISCNTS_LIB")
        if viscnts_lib is not None:
            cli_args.append("-DVISCNTS_LIB=" + str(viscnts_lib))
        cmake.configure(cli_args=cli_args)
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
