using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

// Authoring-only route evidence. No tank, XR, splat renderer or gameplay settings are changed.
public static class TrackRoutePreviewBuilder
{
    private const string Folder = "Assets/Track3DGSData";
    private const string DataPath = Folder + "/route_preview.json";
    private const string ScenePath = "Assets/Scenes/Track3DGS_RoutePreview.unity";

    [Serializable] public class Marker
    {
        public string frameId;
        public float time, distance, x, y, z, fx, fy, fz, ux, uy, uz;
        public int region;
        public bool heldOut;
        public Vector3 Position => new Vector3(x, y, z);
        public Vector3 Forward => new Vector3(fx, fy, fz);
        public Vector3 Up => new Vector3(ux, uy, uz);
    }
    [Serializable] public class Region
    {
        public int index;
        public string name;
        public float coreStart, coreEnd, contextStart, contextEnd, startTime, endTime;
    }
    [Serializable] public class Route
    {
        public int schemaVersion;
        public string routeId, status, scaleNote, coordinateNote, routeSha256, qualityNote;
        public bool qualityPassed;
        public float length, maxGapSeconds;
        public Marker[] markers;
        public Region[] regions;
    }

    private static readonly string[] Palette =
        { "#38bdf8", "#2dd4bf", "#a3e635", "#fbbf24", "#fb923c", "#f472b6", "#c084fc", "#818cf8" };

    [MenuItem("Tools/Track3DGS/Rebuild Route Preview")]
    public static void Rebuild()
    {
        if (!Application.isBatchMode && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (!File.Exists(DataPath)) throw new FileNotFoundException("Copy reports/route_preview.json from Track3DGS first", DataPath);
        Route data = JsonUtility.FromJson<Route>(File.ReadAllText(DataPath));
        if (data.schemaVersion != 1 || data.markers == null || data.markers.Length < 2)
            throw new InvalidDataException("Unsupported or empty route preview");
        foreach (Marker m in data.markers)
            if (!float.IsFinite(m.x) || !float.IsFinite(m.y) || !float.IsFinite(m.z) || m.Forward.sqrMagnitude < .9f || m.Up.sqrMagnitude < .9f)
                throw new InvalidDataException("Invalid route camera: " + m.frameId);

        Directory.CreateDirectory(Folder + "/Materials");
        Directory.CreateDirectory("Assets/Scenes");
        AssetDatabase.Refresh();
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        scene.name = "Track3DGS_RoutePreview";
        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = Color.white;
        var root = new GameObject("Track3DGS | Camera route | APPROXIMATE SCALE");
        var pathRoot = new GameObject("Route segments (gaps are not connected)").transform;
        pathRoot.SetParent(root.transform);
        var cameraRoot = new GameObject("Reconstructed camera positions").transform;
        cameraRoot.SetParent(root.transform);
        var boundaryRoot = new GameObject("Training core boundaries").transform;
        boundaryRoot.SetParent(root.transform);
        var labelRoot = new GameObject("Labels").transform;
        labelRoot.SetParent(root.transform);

        var materials = Palette.Select((hex,i) => MaterialAsset("Region" + i, Parse(hex))).ToArray();
        Material margin = MaterialAsset("CaptureMargin", Parse("#64748b"));
        Material white = MaterialAsset("White", Color.white);
        Material end = MaterialAsset("End", Parse("#fb7185"));
        var bounds = new Bounds(data.markers[0].Position, Vector3.zero);
        for (int i = 0; i < data.markers.Length; i++)
        {
            Marker m = data.markers[i];
            bounds.Encapsulate(m.Position);
            Material mat = m.region < 0 ? margin : materials[m.region % materials.Length];
            var marker = Primitive(PrimitiveType.Sphere,
                $"Camera {i:0000} | {m.time:F2}s | {m.distance:F1}m | {(m.heldOut ? "HELD OUT" : "train")}",
                m.Position, Vector3.one * .8f, mat, cameraRoot);
            marker.transform.rotation = Quaternion.LookRotation(m.Forward, m.Up);
            if (i > 0 && m.time - data.markers[i-1].time <= data.maxGapSeconds)
                Line("Route " + i, data.markers[i-1].Position, m.Position, .65f, mat, pathRoot);
            if (i % 4 == 0)
            {
                Vector3 tip = m.Position + m.Forward * 1.7f;
                Line("Camera forward " + i, m.Position, tip, .12f, mat, cameraRoot);
                Vector3 side = Vector3.Cross(m.Up, m.Forward).normalized * .35f;
                Line("Arrow left " + i, tip, tip - m.Forward * .5f + side, .12f, mat, cameraRoot);
                Line("Arrow right " + i, tip, tip - m.Forward * .5f - side, .12f, mat, cameraRoot);
            }
            if (i > 0 && i % 40 == 0) Label($"{m.time:F0}s / {m.distance:F0}m", m.Position + new Vector3(0, 2, 9), 1.05f, Color.white, labelRoot);
        }

        foreach (Region r in data.regions ?? Array.Empty<Region>())
        {
            Marker m = Closest(data, r.coreStart);
            var index = Array.IndexOf(data.markers, m);
            var tangent = (data.markers[Math.Min(index+1, data.markers.Length-1)].Position - data.markers[Math.Max(0,index-1)].Position).normalized;
            var side = Vector3.Cross(Vector3.up, tangent).normalized;
            if (side.sqrMagnitude < .1f) side = Vector3.right;
            Material mat = materials[r.index % materials.Length];
            Line($"{r.name} CORE START {r.coreStart:F1}m", m.Position-side*8, m.Position+side*8, .4f, mat, boundaryRoot);
            Label($"{r.name}\ncore {r.coreStart:F0}-{r.coreEnd:F0}m\nvideo {r.startTime:F1}-{r.endTime:F1}s",
                m.Position + new Vector3(3, 3, -25), 1.3f, mat.color, labelRoot);
        }
        if (data.regions != null && data.regions.Length > 0)
        {
            float finalCoreEnd = data.regions[data.regions.Length - 1].coreEnd;
            Marker m = Closest(data, finalCoreEnd);
            int index = Array.IndexOf(data.markers, m);
            Vector3 tangent = (data.markers[Math.Min(index+1,data.markers.Length-1)].Position - data.markers[Math.Max(0,index-1)].Position).normalized;
            Vector3 side = Vector3.Cross(Vector3.up,tangent).normalized;
            if (side.sqrMagnitude < .1f) side = Vector3.right;
            Line($"PLAYABLE END {finalCoreEnd:F1}m",m.Position-side*8,m.Position+side*8,.4f,white,boundaryRoot);
            Label($"PLAYABLE END {finalCoreEnd:F1}m",m.Position+new Vector3(-8,3,0),1.3f,Color.white,labelRoot,TextAnchor.MiddleRight);
        }
        var first = data.markers[0];
        var last = data.markers[data.markers.Length-1];
        Primitive(PrimitiveType.Sphere, "CAPTURE START", first.Position, Vector3.one*2, white, boundaryRoot);
        Primitive(PrimitiveType.Sphere, "CAPTURE END", last.Position, Vector3.one*2, end, boundaryRoot);
        Label("CAPTURE START", first.Position + new Vector3(-4,3,12), 1.5f, Color.white, labelRoot,TextAnchor.MiddleRight);
        Label("CAPTURE END", last.Position + new Vector3(-8,3,8), 1.5f, end.color, labelRoot,TextAnchor.MiddleRight);
        Label($"{data.routeId}  /  {data.length:F1} nominal metres\n{data.markers.Length} camera positions  /  {data.regions.Length} proposed training regions\n{data.qualityNote}\n{data.scaleNote}\nCamera path only - not ground geometry",
            new Vector3(bounds.min.x, bounds.max.y + 5, bounds.max.z + 55), 1.65f, Color.white, labelRoot);

        // Fit rendered labels as well as camera centres, so the overview does not
        // clip long region names or place its heading over the capture start.
        foreach (Renderer renderer in root.GetComponentsInChildren<Renderer>())
            bounds.Encapsulate(renderer.bounds);

        var cameraObject = new GameObject("Route Overview Camera");
        Camera camera = cameraObject.AddComponent<Camera>();
        camera.tag = "MainCamera";
        camera.orthographic = true;
        camera.aspect = 16f / 10f;
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = Parse("#0b1220");
        camera.transform.position = new Vector3(bounds.center.x, bounds.max.y + 150, bounds.center.z);
        camera.transform.rotation = Quaternion.Euler(90, 0, 0);
        camera.orthographicSize = Mathf.Max(bounds.extents.z + 15, (bounds.extents.x + 15) / camera.aspect);
        camera.nearClipPlane = .1f;
        camera.farClipPlane = bounds.size.y + 400;
        var extra = camera.GetUniversalAdditionalCameraData();
        extra.renderPostProcessing = false;
        extra.allowXRRendering = false;
        if (!EditorSceneManager.SaveScene(scene, ScenePath)) throw new IOException("Could not save preview scene");
        AssetDatabase.SaveAssets();
        if (SceneView.lastActiveSceneView != null)
            SceneView.lastActiveSceneView.LookAt(bounds.center, Quaternion.Euler(55,-20,0), Mathf.Max(bounds.size.x,bounds.size.z)*.65f);
        RenderPreview(camera);
        File.WriteAllText(Folder + "/scene-build-report.json", JsonUtility.ToJson(new BuildReport {
            scene = ScenePath, markers = data.markers.Length, regions = data.regions.Length,
            routeSha256 = data.routeSha256, sourceStatus = data.status
        }, true));
        AssetDatabase.Refresh();
        Debug.Log($"TRACK_ROUTE_PREVIEW_OK scene={ScenePath} markers={data.markers.Length} regions={data.regions.Length}");
    }

    [Serializable] private class BuildReport { public string scene, routeSha256, sourceStatus; public int markers, regions; }
    private static Marker Closest(Route data, float s) => data.markers.OrderBy(m => Mathf.Abs(m.distance-s)).First();
    private static Color Parse(string hex) { ColorUtility.TryParseHtmlString(hex,out Color c); return c; }

    private static Material MaterialAsset(string name, Color color)
    {
        string path = Folder + "/Materials/" + name + ".mat";
        var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (mat == null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null) throw new InvalidOperationException("URP Unlit shader missing");
            mat = new Material(shader);
            AssetDatabase.CreateAsset(mat,path);
        }
        mat.color = color;
        mat.SetColor("_BaseColor",color);
        EditorUtility.SetDirty(mat);
        return mat;
    }

    private static GameObject Primitive(PrimitiveType type, string name, Vector3 position, Vector3 scale, Material material, Transform parent)
    {
        var go = GameObject.CreatePrimitive(type);
        go.name = name;
        go.transform.SetParent(parent);
        go.transform.position = position;
        go.transform.localScale = scale;
        UnityEngine.Object.DestroyImmediate(go.GetComponent<Collider>());
        go.GetComponent<Renderer>().sharedMaterial = material;
        return go;
    }

    private static void Line(string name, Vector3 a, Vector3 b, float width, Material material, Transform parent)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent);
        var line = go.AddComponent<LineRenderer>();
        line.sharedMaterial = material;
        line.useWorldSpace = true;
        line.positionCount = 2;
        line.SetPositions(new[] {a,b});
        line.startWidth = line.endWidth = width;
        line.numCapVertices = 2;
    }

    private static void Label(string text, Vector3 position, float size, Color color, Transform parent, TextAnchor anchor = TextAnchor.MiddleLeft)
    {
        var go = new GameObject(text.Replace('\n',' '));
        go.transform.SetParent(parent);
        go.transform.position = position;
        go.transform.rotation = Quaternion.Euler(90,0,0);
        var label = go.AddComponent<TextMesh>();
        label.text = text;
        label.characterSize = size;
        label.fontSize = 40;
        label.anchor = anchor;
        label.color = color;
        label.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        go.GetComponent<MeshRenderer>().sharedMaterial = label.font.material;
    }

    private static void RenderPreview(Camera camera)
    {
        var rt = new RenderTexture(2560,1600,24,RenderTextureFormat.ARGB32);
        var image = new Texture2D(2560,1600,TextureFormat.RGB24,false);
        RenderTexture previous = RenderTexture.active;
        try
        {
            if (GraphicsSettings.currentRenderPipeline != null)
                RenderPipeline.SubmitRenderRequest(camera, new UniversalRenderPipeline.SingleCameraRequest { destination = rt });
            else
            {
                camera.targetTexture = rt;
                camera.Render();
            }
            RenderTexture.active = rt;
            image.ReadPixels(new Rect(0,0,2560,1600),0,0);
            image.Apply();
            File.WriteAllBytes(Folder + "/route_scene_overview.png",image.EncodeToPNG());
        }
        finally
        {
            camera.targetTexture = null;
            RenderTexture.active = previous;
            rt.Release();
            UnityEngine.Object.DestroyImmediate(rt);
            UnityEngine.Object.DestroyImmediate(image);
        }
    }
}
