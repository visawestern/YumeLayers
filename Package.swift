// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "YumeLayers",
    platforms: [.macOS(.v26)],
    targets: [
        .testTarget(name: "YumeLayersCoreTests", dependencies: ["YumeLayersCore"]),
        .target(
            name: "YumeLayersCore",
            path: "Sources/YumeLayersCore",
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("ImageIO"),
            ]
        ),
        .executableTarget(
            name: "YumeLayersUI",
            dependencies: ["YumeLayersCore"],
            path: "Sources/YumeLayersUI",
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("AVKit"),
                .linkedFramework("AVFoundation"),
            ]
        )
    ],
    swiftLanguageModes: [.v5]
)
