// Manual test for LoadFromPlyBytes.
// Add to a GameObject together with GsplatRenderer, set PlyPath (absolute, or relative
// to StreamingAssets), and enter Play mode. The PLY is read with File.ReadAllBytes and
// loaded via LoadFromPlyBytes — no importer / asset file involved — then rendered.

using System.IO;
using UnityEngine;

namespace Gsplat
{
    [RequireComponent(typeof(GsplatRenderer))]
    public class RuntimePlyBytesLoaderTest : MonoBehaviour
    {
        [Tooltip("Absolute path, or relative to StreamingAssets")]
        public string PlyPath;

        public CompressionMode Compression = CompressionMode.Spark;
        public SourceCoordinates SourceCoordinates = SourceCoordinates.RUB;

        [Tooltip("Removes splats whose opacity (after sigmoid) is below this value while loading. 0 disables pruning.")]
        [Range(0f, 0.99f)]
        public float OpacityPruneThreshold = 0f;

        void Start()
        {
            var path = Path.IsPathRooted(PlyPath)
                ? PlyPath
                : Path.Combine(Application.streamingAssetsPath, PlyPath);
            if (!File.Exists(path))
            {
                Debug.LogError($"PLY not found: {path}");
                return;
            }

            // Simulates data arriving as a byte array (e.g. downloaded at runtime).
            var bytes = File.ReadAllBytes(path);

            GsplatAsset asset = Compression == CompressionMode.Spark
                ? ScriptableObject.CreateInstance<GsplatAssetSpark>()
                : ScriptableObject.CreateInstance<GsplatAssetUncompressed>();
            asset.LoadFromPlyBytes(bytes, null, SourceCoordinates, OpacityPruneThreshold);

            GetComponent<GsplatRenderer>().GsplatAsset = asset;
            var pruned = asset.PrunedSplatCount;
            var sourceSplatCount = asset.SplatCount + asset.PrunedSplatCount;
            Debug.Log($"Loaded {asset.SplatCount:N0} splats (SH bands {asset.SHBands}) " +
                      $"from {bytes.Length:N0} bytes via LoadFromPlyBytes" +
                      (pruned > 0
                          ? $", pruned {pruned:N0} / {sourceSplatCount:N0} splats " +
                            $"({pruned * 100.0 / sourceSplatCount:F1}%) below opacity {OpacityPruneThreshold}"
                          : ""));
        }
    }
}