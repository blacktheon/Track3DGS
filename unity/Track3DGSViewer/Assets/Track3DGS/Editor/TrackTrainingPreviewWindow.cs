using System;
using System.IO;
using System.Linq;
using Gsplat;
using Gsplat.Editor;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering.Universal;

/// <summary>Editor review of trained regions. Only two scene renderer slots exist.</summary>
public sealed class TrackTrainingPreviewWindow : EditorWindow
{
    public const string ScenePath = "Assets/Scenes/Track3DGS_TrainingPreview.unity";
    public const string CatalogPath = "Assets/Track3DGSData/catalog.json";
    const string RootName = "Track3DGS Models | Maximum two visible";
    const string CameraName = "Training Preview Camera";
    const string RoutePath = "Assets/Track3DGSData/route_preview.json";

    [Serializable] public class Entry
    {
        public int index, rawCount, cleanCount, coreCount, skyCount;
        public string id, rawPath, cleanPath, corePath, skyPath, skyReport, status, sourceModel;
        public float coreStart, coreEnd, x, y, z;
        public Vector3 Origin => new Vector3(x, y, z);
        public string PathFor(string variant) => variant == "raw" ? rawPath : variant == "clean" ? cleanPath : variant == "sky" ? skyPath : corePath;
        public int CountFor(string variant) => variant == "raw" ? rawCount : variant == "clean" ? cleanCount : variant == "sky" ? skyCount : coreCount;
    }
    [Serializable] public class ReviewView
    {
        public string label;
        public int cellIndex;
        public float station, fieldOfView;
        public Vector3 position, forward, up;
    }
    [Serializable] public class Catalog { public int schemaVersion; public string routeSha256, note; public Entry[] entries; public ReviewView[] reviewViews; }

    [SerializeField] int selectedIndex, variantIndex = 2;
    [SerializeField] bool includeNext, uncompressed, lookBack;
    [SerializeField] float station = 70;
    [SerializeField] int selectedReviewView;
    [SerializeField] Vector2 scrollPosition;
    string lastMessage = "Training outputs will appear as each chunk finishes.";
    static readonly string[] Variants = { "raw", "clean", "core", "sky" };

    [MenuItem("Tools/Track3DGS/Training Model Preview")]
    public static void Open() => GetWindow<TrackTrainingPreviewWindow>("Track3DGS Models");

    public static Catalog ReadCatalog()
    {
        if (!File.Exists(CatalogPath)) return new Catalog { schemaVersion = 1, entries = Array.Empty<Entry>() };
        var catalog = JsonUtility.FromJson<Catalog>(File.ReadAllText(CatalogPath));
        if (catalog.schemaVersion != 1 || catalog.entries == null) throw new InvalidDataException("Unsupported model catalog");
        if (File.Exists(RoutePath))
        {
            var route = JsonUtility.FromJson<TrackRoutePreviewBuilder.Route>(File.ReadAllText(RoutePath));
            if (route.routeSha256 != catalog.routeSha256) throw new InvalidDataException("Catalog and route markers belong to different revisions.");
        }
        return catalog;
    }

    void OnGUI()
    {
        using (var scroll = new EditorGUILayout.ScrollViewScope(scrollPosition))
        {
            scrollPosition = scroll.scrollPosition;
            DrawControls();
        }
    }

    void DrawControls()
    {
        EditorGUILayout.LabelField("Regional training", EditorStyles.boldLabel);
        EditorGUILayout.HelpBox("Draft reconstruction. Scale and absolute grade are approximate. The scene has two model slots; switching clears both before loading.", MessageType.Info);
        var catalog = ReadCatalog();
        EditorGUILayout.LabelField($"Completed outputs: {catalog.entries.Length}");
        if (GUILayout.Button("Open / create training scene")) TryAction(EnsureScene);
        if (catalog.entries.Length > 0)
        {
            selectedIndex = Mathf.Clamp(selectedIndex, 0, catalog.entries.Length - 1);
            selectedIndex = EditorGUILayout.Popup("First chunk", selectedIndex, catalog.entries.Select(e => $"{e.index + 1}: {e.id} ({e.coreStart:F0}–{e.coreEnd:F0} m)").ToArray());
            variantIndex = EditorGUILayout.Popup("Representation", variantIndex, new[] { "Original training result", "Cleaned, with context", "Trimmed route core", "Sky cleaned core (experimental)" });
            includeNext = EditorGUILayout.Toggle("Show next neighbour", includeNext);
            uncompressed = EditorGUILayout.Toggle("Uncompressed reference", uncompressed);
            if (includeNext && uncompressed)
                EditorGUILayout.HelpBox("Uncompressed reference supports one chunk. Adjacent pairs use Spark for global depth sorting; full-precision pair renders are in the Track3DGS QC report.", MessageType.Info);
            var entry = catalog.entries[selectedIndex];
            EditorGUILayout.LabelField("Selected model", $"{entry.CountFor(Variants[variantIndex]):N0} splats");
            if (includeNext && variantIndex < 2)
                EditorGUILayout.HelpBox("Raw and cleaned context regions intentionally overlap. Use trimmed route cores to inspect the assembled seam.", MessageType.Warning);
            var next = catalog.entries.FirstOrDefault(e => e.index == entry.index + 1);
            bool hasSelected = HasVariant(entry, Variants[variantIndex]) && (!includeNext || HasVariant(next, Variants[variantIndex]));
            using (new EditorGUI.DisabledScope(!hasSelected))
            if (GUILayout.Button("Load selected model(s)")) TryAction(() =>
            {
                LoadModels(entry.index, Variants[variantIndex], includeNext, uncompressed);
                station = includeNext ? entry.coreEnd : (entry.coreStart + entry.coreEnd) * .5f;
                MoveCamera(station, lookBack);
                lastMessage = $"Loaded {entry.id}, {Variants[variantIndex]}. At most two visible.";
            });
            if (!hasSelected) EditorGUILayout.HelpBox("This representation is not available for every selected chunk.", MessageType.Info);
            bool hasSky = HasVariant(entry, "sky") && (!includeNext || HasVariant(next, "sky"));
            using (new EditorGUI.DisabledScope(!hasSky))
            {
                EditorGUILayout.LabelField("Sky cleanup comparison", EditorStyles.boldLabel);
                using (new EditorGUILayout.HorizontalScope())
                {
                    if (GUILayout.Button("Before: original core")) TryAction(() => LoadModels(entry.index, "core", includeNext, uncompressed));
                    if (GUILayout.Button("After: sky cleaned")) TryAction(() => LoadModels(entry.index, "sky", includeNext, uncompressed));
                }
            }
            if (hasSky)
            {
                EditorGUILayout.LabelField("Core reduction", $"{entry.coreCount:N0} → {entry.skyCount:N0} ({100f * (entry.coreCount - entry.skyCount) / entry.coreCount:F2}%)");
                EditorGUILayout.HelpBox("Before/After replaces the loaded assets at the same camera pose. Sky cleanup is experimental; some vegetation and ground artifacts remain.", MessageType.Info);
                if (File.Exists(entry.skyReport) && GUILayout.Button("Open cleanup comparison report"))
                    EditorUtility.OpenWithDefaultApp(entry.skyReport);
            }
            var reviewViews = (catalog.reviewViews ?? Array.Empty<ReviewView>())
                .Select((view, index) => new { view, index })
                .Where(v => v.view.cellIndex == entry.index || (includeNext && v.view.cellIndex == entry.index + 1)).ToArray();
            if (reviewViews.Length > 0)
            {
                selectedReviewView = Mathf.Clamp(selectedReviewView, 0, reviewViews.Length - 1);
                selectedReviewView = EditorGUILayout.Popup("Recorded camera view", selectedReviewView, reviewViews.Select(v => v.view.label).ToArray());
                if (GUILayout.Button("Move to recorded camera")) TryAction(() => MoveToReviewView(reviewViews[selectedReviewView].index));
            }
            // Recorded cameras can lie anywhere in either selected core. A
            // short seam-only range would clamp a Model 2 pose on the next repaint.
            float hi = includeNext && next != null ? next.coreEnd : entry.coreEnd;
            float nextStation = EditorGUILayout.Slider("Camera route station", station, entry.coreStart, hi);
            bool nextLookBack = EditorGUILayout.Toggle("Look back along route", lookBack);
            if (!Mathf.Approximately(nextStation, station) || nextLookBack != lookBack)
            {
                station = nextStation; lookBack = nextLookBack;
                TryAction(() => MoveCamera(station, lookBack));
            }
            if (GUILayout.Button("Use Scene view position for Game camera")) TryAction(UseSceneView);
            if (GUILayout.Button("Show native output folder")) EditorUtility.RevealInFinder(entry.sourceModel);
        }
        if (GUILayout.Button("Unload both models")) TryAction(() => { ClearSlots(); lastMessage = "Both model slots unloaded."; });
        EditorGUILayout.HelpBox(lastMessage, MessageType.None);
    }

    static bool HasVariant(Entry entry, string variant) => entry != null && entry.CountFor(variant) > 0 && File.Exists(entry.PathFor(variant));

    void TryAction(Action action)
    {
        try { action(); }
        catch (Exception e) { lastMessage = e.Message; Debug.LogException(e); }
    }

    public static void EnsureScene()
    {
        if (EditorApplication.isPlaying) throw new InvalidOperationException("Stop Play mode before changing training previews.");
        if (UnityEngine.SceneManagement.SceneManager.GetActiveScene().path == ScenePath) return;
        if (UnityEngine.SceneManagement.SceneManager.GetActiveScene().isDirty)
            throw new InvalidOperationException("Save the current scene before opening the training scene.");
        if (File.Exists(ScenePath)) { EditorSceneManager.OpenScene(ScenePath); return; }
        Directory.CreateDirectory("Assets/Scenes");
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        var root = new GameObject(RootName);
        for (int i = 0; i < 2; i++)
        {
            var slot = new GameObject("Slot " + (i + 1)); slot.transform.SetParent(root.transform);
            slot.SetActive(false); slot.AddComponent<GsplatRenderer>();
        }
        var camera = new GameObject(CameraName).AddComponent<Camera>();
        camera.tag = "MainCamera"; camera.fieldOfView = 85; camera.nearClipPlane = .05f; camera.farClipPlane = 500;
        camera.clearFlags = CameraClearFlags.SolidColor; camera.backgroundColor = new Color(.16f, .23f, .31f);
        camera.allowHDR = false; camera.allowMSAA = false;
        var urp = camera.GetUniversalAdditionalCameraData(); urp.renderPostProcessing = false; urp.allowXRRendering = false;
        MoveCamera(70, false);
        if (!EditorSceneManager.SaveScene(scene, ScenePath)) throw new IOException("Could not save training scene");
    }

    static GsplatRenderer[] GetSlots()
    {
        if (UnityEngine.SceneManagement.SceneManager.GetActiveScene().path != ScenePath)
            throw new InvalidOperationException("Open the training preview scene first.");
        var root = GameObject.Find(RootName);
        if (!root) throw new InvalidOperationException("Training model slots are missing.");
        var slots = root.GetComponentsInChildren<GsplatRenderer>(true);
        var all = UnityEngine.Object.FindObjectsByType<GsplatRenderer>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        if (slots.Length != 2 || all.Length != 2) throw new InvalidOperationException("Training scene must contain exactly two Gaussian renderer slots.");
        return slots;
    }

    public static void ClearSlots()
    {
        foreach (var renderer in GetSlots())
        {
            renderer.gameObject.SetActive(false); renderer.GsplatAsset = null;
            EditorUtility.SetDirty(renderer);
        }
        EditorUtility.UnloadUnusedAssetsImmediate();
        EditorSceneManager.MarkSceneDirty(UnityEngine.SceneManagement.SceneManager.GetActiveScene());
    }

    public static void LoadModels(int firstIndex, string variant = "core", bool includeNeighbour = false, bool uncompressedReference = false)
    {
        if (!Variants.Contains(variant)) throw new ArgumentException("Unknown representation");
        if (includeNeighbour && uncompressedReference)
            throw new InvalidOperationException("Uncompressed reference supports one chunk; use Spark for globally sorted pairs.");
        EnsureScene();
        var catalog = ReadCatalog();
        var first = catalog.entries.FirstOrDefault(e => e.index == firstIndex);
        if (first == null) throw new InvalidOperationException("This chunk has not finished processing.");
        var second = includeNeighbour ? catalog.entries.FirstOrDefault(e => e.index == firstIndex + 1) : null;
        if (includeNeighbour && second == null) throw new InvalidOperationException("The next chunk has not finished processing.");
        var entries = second == null ? new[] { first } : new[] { first, second };
        foreach (var entry in entries)
            if (!File.Exists(entry.PathFor(variant))) throw new FileNotFoundException("Missing model cache", entry.PathFor(variant));
        ClearSlots();
        var slots = GetSlots();
        try
        {
            for (int i = 0; i < entries.Length; i++)
            {
                var entry = entries[i]; string path = entry.PathFor(variant);
                AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceSynchronousImport);
                var importer = AssetImporter.GetAtPath(path) as GsplatImporter;
                if (!importer) throw new InvalidDataException("Gaussian PLY importer is not installed");
                var compression = uncompressedReference ? CompressionMode.Uncompressed : CompressionMode.Spark;
                if (importer.Compression != compression || importer.SourceCoordinates != SourceCoordinates.RUB || importer.OpacityPruneThreshold != 0)
                {
                    importer.Compression = compression; importer.SourceCoordinates = SourceCoordinates.RUB;
                    importer.OpacityPruneThreshold = 0; importer.SaveAndReimport();
                }
                var asset = AssetDatabase.LoadAssetAtPath<GsplatAsset>(path);
                if (!asset || asset.SplatCount != entry.CountFor(variant)) throw new InvalidDataException("Imported splat count differs from the source");
                var slot = slots[i]; slot.name = $"Slot {i + 1} | {entry.id} | {variant}";
                slot.transform.SetPositionAndRotation(entry.Origin, Quaternion.identity); slot.transform.localScale = Vector3.one;
                // Use Unity serialization so the renderer's OnValidate refreshes
                // its cached asset GUID, which the importer uses after reimports.
                var binding = new SerializedObject(slot);
                binding.FindProperty("GsplatAsset").objectReferenceValue = asset;
                binding.ApplyModifiedPropertiesWithoutUndo();
                slot.SHDegree = 3; slot.GammaToLinear = QualitySettings.activeColorSpace == ColorSpace.Linear;
                slot.AsyncUpload = true; slot.RenderBeforeUploadComplete = false; slot.Brightness = 1;
                slot.gameObject.SetActive(true); EditorUtility.SetDirty(slot);
            }
        }
        catch { ClearSlots(); throw; }
        GsplatSettings.Instance.EnableGlobalSort = true;
        EditorUtility.SetDirty(GsplatSettings.Instance);
        EditorSceneManager.MarkSceneDirty(UnityEngine.SceneManagement.SceneManager.GetActiveScene());
        EditorSceneManager.SaveScene(UnityEngine.SceneManagement.SceneManager.GetActiveScene());
        AssetDatabase.SaveAssets(); SceneView.RepaintAll(); EditorApplication.QueuePlayerLoopUpdate();
        foreach (var window in Resources.FindObjectsOfTypeAll<TrackTrainingPreviewWindow>())
        {
            window.selectedIndex = Array.FindIndex(catalog.entries, e => e.index == firstIndex);
            window.variantIndex = Array.IndexOf(Variants, variant);
            window.includeNext = includeNeighbour; window.uncompressed = uncompressedReference;
            window.lastMessage = $"Loaded {first.id}, {variant}; {entries.Length} visible model(s).";
            window.Repaint();
        }
    }

    public static void MoveCamera(float station, bool backwards)
    {
        var go = GameObject.Find(CameraName); if (!go || !File.Exists(RoutePath)) return;
        var route = JsonUtility.FromJson<TrackRoutePreviewBuilder.Route>(File.ReadAllText(RoutePath));
        var samples = route.markers;
        int index = Array.FindIndex(samples, m => m.distance >= station);
        index = Mathf.Clamp(index < 0 ? samples.Length - 1 : index, 1, samples.Length - 1);
        var a = samples[index - 1]; var b = samples[index];
        float t = Mathf.InverseLerp(a.distance, b.distance, station);
        var position = Vector3.Lerp(a.Position, b.Position, t);
        var direction = b.Position - a.Position;
        if (direction.sqrMagnitude < .0001f) direction = b.Forward;
        if (backwards) direction = -direction;
        go.transform.SetPositionAndRotation(position, Quaternion.LookRotation(direction, Vector3.up));
        if (SceneView.lastActiveSceneView) SceneView.lastActiveSceneView.LookAtDirect(position + direction.normalized * 5, go.transform.rotation, 5);
        EditorSceneManager.MarkSceneDirty(go.scene); EditorApplication.QueuePlayerLoopUpdate();
        foreach (var window in Resources.FindObjectsOfTypeAll<TrackTrainingPreviewWindow>())
        {
            window.station = station; window.lookBack = backwards; window.Repaint();
        }
    }

    public static void MoveToReviewView(int index)
    {
        if (UnityEngine.SceneManagement.SceneManager.GetActiveScene().path != ScenePath)
            throw new InvalidOperationException("Load the comparison models first.");
        var catalog = ReadCatalog();
        if (catalog.reviewViews == null || index < 0 || index >= catalog.reviewViews.Length)
            throw new ArgumentOutOfRangeException(nameof(index));
        var view = catalog.reviewViews[index];
        var go = GameObject.Find(CameraName);
        if (!go) throw new InvalidOperationException("Training preview camera is missing.");
        go.transform.SetPositionAndRotation(view.position, Quaternion.LookRotation(view.forward, view.up));
        var camera = go.GetComponent<Camera>();
        camera.fieldOfView = view.fieldOfView; camera.clearFlags = CameraClearFlags.SolidColor; camera.backgroundColor = Color.black;
        if (SceneView.lastActiveSceneView)
            SceneView.lastActiveSceneView.LookAtDirect(view.position + view.forward.normalized * 5, go.transform.rotation, 5);
        foreach (var window in Resources.FindObjectsOfTypeAll<TrackTrainingPreviewWindow>())
        {
            window.station = view.station;
            var available = catalog.reviewViews.Where(v => v.cellIndex == catalog.entries[window.selectedIndex].index ||
                (window.includeNext && v.cellIndex == catalog.entries[window.selectedIndex].index + 1)).ToArray();
            int local = Array.IndexOf(available, view);
            if (local >= 0) window.selectedReviewView = local;
            window.lastMessage = "Recorded camera: " + view.label; window.Repaint();
        }
        EditorSceneManager.MarkSceneDirty(go.scene); EditorApplication.QueuePlayerLoopUpdate();
    }

    static void UseSceneView()
    {
        var view = SceneView.lastActiveSceneView; var go = GameObject.Find(CameraName);
        if (!view || !go) return;
        go.transform.SetPositionAndRotation(view.camera.transform.position, view.camera.transform.rotation);
        EditorSceneManager.MarkSceneDirty(go.scene); EditorApplication.QueuePlayerLoopUpdate();
    }
}
