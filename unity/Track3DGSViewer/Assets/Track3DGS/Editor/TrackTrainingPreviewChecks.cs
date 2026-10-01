using System;
using System.IO;
using System.Linq;
using System.Text;
using Gsplat;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.SceneManagement;

public static class TrackTrainingPreviewChecks
{
    public static string CheckSkyCatalog()
    {
        var entry = JsonUtility.FromJson<TrackTrainingPreviewWindow.Entry>(
            "{\"rawPath\":\"raw.ply\",\"cleanPath\":\"clean.ply\",\"corePath\":\"core.ply\",\"skyPath\":\"sky.ply\",\"rawCount\":100,\"cleanCount\":90,\"coreCount\":80,\"skyCount\":70}");
        string[] variants = { "raw", "clean", "core", "sky" };
        int[] counts = { 100, 90, 80, 70 };
        for (int i = 0; i < variants.Length; i++)
            if (entry.PathFor(variants[i]) != variants[i] + ".ply" || entry.CountFor(variants[i]) != counts[i])
                return $"FAIL: {variants[i]} returned {entry.PathFor(variants[i])}, {entry.CountFor(variants[i])}; expected {variants[i]}.ply, {counts[i]}";
        return "PASS: raw / clean / core / sky paths and counts remain distinct.";
    }

    public static string CheckSkyComparison()
    {
        TrackTrainingPreviewWindow.EnsureScene();
        const string folder = "Library/TrackStep2Checks/SkyCleanup";
        Directory.CreateDirectory(folder);
        var camera = GameObject.Find("Training Preview Camera").GetComponent<Camera>();
        var log = new StringBuilder(CheckSkyCatalog()).AppendLine();
        for (int index = 0; index < 2; index++)
        {
            TrackTrainingPreviewWindow.LoadModels(index, "core", false, false);
            TrackTrainingPreviewWindow.MoveToReviewView(Array.FindIndex(TrackTrainingPreviewWindow.ReadCatalog().reviewViews, v => v.cellIndex == index));
            VerifySkySlots(index, "core", false);
            var position = camera.transform.position; var rotation = camera.transform.rotation; float fov = camera.fieldOfView;
            CaptureSkyPng(camera, folder + $"/model{index + 1}_before.png");
            TrackTrainingPreviewWindow.LoadModels(index, "sky", false, false);
            VerifySkySlots(index, "sky", false);
            if (camera.transform.position != position || Quaternion.Angle(camera.transform.rotation, rotation) > .001f || !Mathf.Approximately(camera.fieldOfView, fov))
                throw new InvalidOperationException("Before/After changed the camera pose or lens.");
            CaptureSkyPng(camera, folder + $"/model{index + 1}_after.png");
            log.AppendLine($"PASS: Model {index + 1} switches original core → sky core with unchanged camera, correct asset count, GUID and position.");
            TrackTrainingPreviewWindow.ClearSlots();
            foreach (var slot in UnityEngine.Object.FindObjectsByType<GsplatRenderer>(FindObjectsInactive.Include, FindObjectsSortMode.None))
                if (slot.SplatCount != 0 || slot.GsplatResource != null || slot.GsplatAsset != null)
                    throw new InvalidOperationException("Unloaded slot retains an asset or GPU resources.");
            log.AppendLine("PASS: unloading releases both assets and GPU resources.");
        }
        TrackTrainingPreviewWindow.LoadModels(0, "sky", true, false);
        TrackTrainingPreviewWindow.MoveToReviewView(0);
        VerifySkySlots(0, "sky", true);
        bool missingRejected = false;
        try { TrackTrainingPreviewWindow.LoadModels(int.MaxValue, "sky", false, false); }
        catch (InvalidOperationException) { missingRejected = true; }
        if (!missingRejected) throw new InvalidOperationException("Unavailable sky representation was accepted.");
        VerifySkySlots(0, "sky", true);
        log.AppendLine("PASS: unavailable sky result rejected before clearing the current models.");
        log.AppendLine("PASS: final scene contains exactly two loaded sky cores in their original route positions.");
        EditorSceneManager.SaveScene(SceneManager.GetActiveScene());
        File.WriteAllText(folder + "/checks.txt", log.ToString());
        return log.ToString();
    }

    static void VerifySkySlots(int firstIndex, string variant, bool includeNext)
    {
        var all = UnityEngine.Object.FindObjectsByType<GsplatRenderer>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        var active = all.Where(r => r.isActiveAndEnabled).ToArray();
        if (all.Length != 2 || active.Length != (includeNext ? 2 : 1))
            throw new InvalidOperationException("Preview exceeds its two-slot budget or has the wrong active count.");
        var catalog = TrackTrainingPreviewWindow.ReadCatalog();
        foreach (var renderer in active)
        {
            var path = AssetDatabase.GetAssetPath(renderer.GsplatAsset);
            var entry = catalog.entries.FirstOrDefault(e => e.PathFor(variant) == path);
            if (entry == null || entry.index < firstIndex || entry.index > firstIndex + (includeNext ? 1 : 0))
                throw new InvalidOperationException("Wrong model/variant bound to a preview slot.");
            renderer.PrepareEditorPreview();
            if (renderer.GsplatAsset.SplatCount != entry.CountFor(variant) || renderer.SplatCount != entry.CountFor(variant) ||
                renderer.transform.position != entry.Origin || renderer.transform.localScale != Vector3.one ||
                Quaternion.Angle(renderer.transform.rotation, Quaternion.identity) > .001f ||
                renderer.AssetGuid != AssetDatabase.AssetPathToGUID(path))
                throw new InvalidOperationException("Imported count, route transform, upload, or serialized asset GUID differs from the catalog.");
        }
        foreach (var inactive in all.Except(active))
            if (inactive.SplatCount != 0 || inactive.GsplatResource != null)
                throw new InvalidOperationException("Inactive preview renderer retains GPU resources.");
    }

    static void CaptureSkyPng(Camera camera, string path)
    {
        var target = RenderTexture.GetTemporary(1024, 1024, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        var previous = RenderTexture.active;
        var texture = new Texture2D(1024, 1024, TextureFormat.RGBA32, false);
        float aspect = camera.aspect;
        try
        {
            camera.aspect = 1;
            RenderPipeline.SubmitRenderRequest(camera, new UniversalRenderPipeline.SingleCameraRequest { destination = target });
            RenderTexture.active = target;
            texture.ReadPixels(new Rect(0, 0, 1024, 1024), 0, 0); texture.Apply();
            File.WriteAllBytes(path, texture.EncodeToPNG());
        }
        finally
        {
            camera.aspect = aspect; RenderTexture.active = previous;
            UnityEngine.Object.DestroyImmediate(texture); RenderTexture.ReleaseTemporary(target);
        }
    }

    // Tiny interleaved-depth scene: two assets must agree with one combined asset.
    // Fixtures live in Library and no production scene references are saved.
    public static string CheckEditorMerge()
    {
        TrackTrainingPreviewWindow.EnsureScene();
        var slots = UnityEngine.Object.FindObjectsByType<GsplatRenderer>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        if (slots.Length != 2 || slots.Any(s => s.GsplatAsset)) throw new InvalidOperationException("Unload both preview slots before the check.");
        var camera = GameObject.Find("Training Preview Camera").GetComponent<Camera>();
        var position = camera.transform.position; var rotation = camera.transform.rotation;
        var background = camera.backgroundColor; var fov = camera.fieldOfView;
        var assets = new GsplatAssetSpark[3];
        try
        {
            WriteFixtures();
            for (int i = 0; i < 3; i++)
            {
                assets[i] = ScriptableObject.CreateInstance<GsplatAssetSpark>();
                assets[i].name = "Depth fixture " + i;
                assets[i].LoadFromPly("Library/TrackStep2Checks/" + new[] { "a", "b", "combined" }[i] + ".ply", null, SourceCoordinates.RUB, 0);
            }
            camera.fieldOfView = 60; camera.backgroundColor = Color.black;
            float worst = 0; bool merged = true;
            for (int direction = 0; direction < 2; direction++)
            {
                camera.transform.SetPositionAndRotation(direction == 0 ? Vector3.zero : new Vector3(0, 0, 7),
                    direction == 0 ? Quaternion.identity : Quaternion.Euler(0, 180, 0));
                Assign(slots[0], assets[0]); Assign(slots[1], assets[1]);
                var pair = Capture(camera);
                if (!pair.Any(c => Mathf.Max(c.r,c.g,c.b) > .1f)) throw new InvalidOperationException("Pair fixture rendered black; a blank image is not a passing sort check.");
                merged &= GsplatSorter.Instance.GlobalRenderEnabled;
                slots[0].gameObject.SetActive(false); slots[1].gameObject.SetActive(false);
                Assign(slots[0], assets[2]);
                var reference = Capture(camera);
                for (int y = 24; y < 40; y++) for (int x = 24; x < 40; x++)
                {
                    var a = pair[y * 64 + x]; var b = reference[y * 64 + x];
                    worst = Mathf.Max(worst, Mathf.Abs(a.r - b.r), Mathf.Abs(a.g - b.g), Mathf.Abs(a.b - b.b));
                }
                slots[0].gameObject.SetActive(false);
            }
            string result = $"Global editor merge: enabled={merged}; pair-vs-combined max RGB error={worst:F6}; forward and reverse views";
            File.WriteAllText("Library/TrackStep2Checks/result.txt", result);
            if (!merged || worst > .015f) throw new InvalidOperationException(result);
            return result;
        }
        finally
        {
            foreach (var slot in slots) { slot.gameObject.SetActive(false); slot.GsplatAsset = null; slot.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity); }
            foreach (var asset in assets) if (asset) UnityEngine.Object.DestroyImmediate(asset);
            camera.transform.SetPositionAndRotation(position, rotation); camera.backgroundColor = background; camera.fieldOfView = fov;
        }
    }

    static void WriteFixtures()
    {
        const string folder = "Library/TrackStep2Checks";
        Directory.CreateDirectory(folder);
        string[] fields = { "x", "y", "z", "opacity", "rot_0", "rot_1", "rot_2", "rot_3",
            "scale_0", "scale_1", "scale_2", "f_dc_0", "f_dc_1", "f_dc_2" };
        var colors = new[] { new Vector3(.9f, .1f, .1f), new Vector3(.1f, .1f, .9f),
            new Vector3(.1f, .9f, .1f), new Vector3(.9f, .9f, .1f) };
        var groups = new[] { new[] { 0, 2 }, new[] { 1, 3 }, new[] { 0, 1, 2, 3 } };
        string[] names = { "a", "b", "combined" };
        for (int group = 0; group < groups.Length; group++)
        {
            var header = new StringBuilder("ply\nformat binary_little_endian 1.0\nelement vertex ")
                .Append(groups[group].Length).Append('\n');
            foreach (var field in fields) header.Append("property float ").Append(field).Append('\n');
            for (int i = 0; i < 45; i++) header.Append("property float f_rest_").Append(i).Append('\n');
            header.Append("end_header\n");
            using var writer = new BinaryWriter(File.Create(folder + "/" + names[group] + ".ply"));
            writer.Write(Encoding.ASCII.GetBytes(header.ToString()));
            foreach (int index in groups[group])
            {
                var row = new float[59];
                row[2] = -(index + 2); row[4] = 1;
                row[8] = row[9] = row[10] = Mathf.Log(.4f);
                for (int channel = 0; channel < 3; channel++)
                    row[11 + channel] = (colors[index][channel] - .5f) / .2820947917f;
                foreach (float value in row) writer.Write(value);
            }
        }
    }

    static void Assign(GsplatRenderer slot, GsplatAsset asset)
    {
        slot.gameObject.SetActive(false); slot.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
        slot.transform.localScale = Vector3.one; slot.GsplatAsset = asset; slot.AsyncUpload = false;
        slot.GammaToLinear = false; slot.SHDegree = 3; slot.gameObject.SetActive(true);
    }

    static Color[] Capture(Camera camera)
    {
        var rt = RenderTexture.GetTemporary(64, 64, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
        var previous = RenderTexture.active;
        var texture = new Texture2D(64, 64, TextureFormat.RGBA32, false, true);
        try
        {
            RenderPipeline.SubmitRenderRequest(camera, new UniversalRenderPipeline.SingleCameraRequest { destination = rt });
            RenderTexture.active = rt; texture.ReadPixels(new Rect(0, 0, 64, 64), 0, 0); texture.Apply();
            return texture.GetPixels();
        }
        finally { RenderTexture.active = previous; UnityEngine.Object.DestroyImmediate(texture); RenderTexture.ReleaseTemporary(rt); }
    }
}
