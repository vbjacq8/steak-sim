// Debug viewer: voxelizes an .obj on the GPU and draws every non-empty voxel as a box outline.
// The side panel is the color key and has the same toggles as the keyboard shortcuts.
//
// Usage: voxel_viewer [model.obj] [voxelSize]
//
// Keys:
//   1        toggle inside-only voxels      (flags == VOXEL_INSIDE)
//   2        toggle surface-only voxels     (flags == VOXEL_SURFACE)
//   3        toggle inside + surface voxels (flags == both)
//   M        toggle the mesh's triangle edges
//   Up/Down  show a single Z slice and move it
//   A        show all slices again

#include "compute_context.hpp"
#include "voxelizer.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <open3d/Open3D.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace gui = open3d::visualization::gui;
namespace rendering = open3d::visualization::rendering;

namespace {

// Boxes are drawn slightly smaller than the voxel so neighboring outlines stay distinguishable.
constexpr double BOX_SHRINK = 0.85;

struct Mesh{
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
};

Mesh loadObj(const std::string& path){
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str())) throw std::runtime_error(warn + err);

    Mesh mesh;
    for (size_t i = 0; i + 2 < attrib.vertices.size(); i += 3){
        mesh.positions.push_back({attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2]});
    }
    for (const auto& shape : shapes){
        for (const auto& index : shape.mesh.indices) mesh.indices.push_back(static_cast<uint32_t>(index.vertex_index));
    }
    return mesh;
}

Eigen::Vector3d toEigen(glm::vec3 v){ return {v.x, v.y, v.z}; }

void appendBox(open3d::geometry::LineSet& lines, Eigen::Vector3d lo, Eigen::Vector3d hi, const Eigen::Vector3d& color){
    static constexpr std::array<std::array<int, 2>, 12> edges{{
        {0, 1}, {2, 3}, {4, 5}, {6, 7},  // along x
        {0, 2}, {1, 3}, {4, 6}, {5, 7},  // along y
        {0, 4}, {1, 5}, {2, 6}, {3, 7},  // along z
    }};
    const int base = static_cast<int>(lines.points_.size());
    for (int corner = 0; corner < 8; ++corner){
        lines.points_.emplace_back((corner & 1) ? hi.x() : lo.x(), (corner & 2) ? hi.y() : lo.y(), (corner & 4) ? hi.z() : lo.z());
    }
    for (const auto& [a, b] : edges){
        lines.lines_.emplace_back(base + a, base + b);
        lines.colors_.push_back(color);
    }
}

// One toggleable group of geometry with a row in the key.
struct Layer{
    std::string name;
    Eigen::Vector3d color;
    uint32_t flags = 0;  // voxel value drawn by this layer; unused for the mesh layer
    bool visible = true;
    open3d::geometry::LineSet lines;
    std::shared_ptr<gui::Checkbox> checkbox;
};

class ViewerWindow : public gui::Window{
public:
    ViewerWindow(const std::string& title, Mesh meshIn, VoxelGrid gridIn)
        : gui::Window(title, 1400, 850), mesh(std::move(meshIn)), grid(std::move(gridIn)){
        layers = {{
            {"inside only", {0.20, 0.45, 0.95}, VOXEL_INSIDE},
            {"surface only", {1.00, 0.60, 0.10}, VOXEL_SURFACE},
            {"inside + surface", {0.90, 0.15, 0.15}, VOXEL_INSIDE | VOXEL_SURFACE},
            {"mesh edges", {0.55, 0.55, 0.55}},
        }};
        buildMeshEdges();

        scene = std::make_shared<gui::SceneWidget>();
        scene->SetScene(std::make_shared<rendering::Open3DScene>(GetRenderer()));
        scene->GetScene()->SetBackground({0.08f, 0.08f, 0.10f, 1.0f});
        material.shader = "unlitLine";
        material.line_width = 1.0f;
        for (size_t i = 0; i < VOXEL_LAYERS; ++i) rebuildVoxels(layers[i]);
        for (Layer& layer : layers) upload(layer);

        const auto& bounds = scene->GetScene()->GetBoundingBox();
        scene->SetupCamera(60.0f, bounds, bounds.GetCenter().cast<float>());
        AddChild(scene);
        AddChild(buildPanel());

        SetOnKeyEvent([this](const gui::KeyEvent& e){ return onKey(e); });
    }

    void Layout(const gui::LayoutContext& context) override{
        const gui::Rect r = GetContentRect();
        const int panelWidth = std::min(r.width / 3, 20 * context.theme.font_size);
        panel->SetFrame(gui::Rect(r.GetRight() - panelWidth, r.y, panelWidth, r.height));
        scene->SetFrame(gui::Rect(r.x, r.y, r.width - panelWidth, r.height));
        gui::Window::Layout(context);
    }

private:
    static constexpr size_t VOXEL_LAYERS = 3;
    static constexpr size_t MESH_LAYER = 3;

    void buildMeshEdges(){
        auto& lines = layers[MESH_LAYER].lines;
        for (const glm::vec3& p : mesh.positions) lines.points_.push_back(toEigen(p));
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3){
            for (int e = 0; e < 3; ++e){
                lines.lines_.emplace_back(mesh.indices[t + e], mesh.indices[t + (e + 1) % 3]);
            }
        }
        lines.PaintUniformColor(layers[MESH_LAYER].color);
    }

    // Uses the current slice; slice < 0 means every Z layer.
    void rebuildVoxels(Layer& layer){
        layer.lines.Clear();
        const double s = grid.voxelSize;
        const Eigen::Vector3d inset = Eigen::Vector3d::Constant(0.5 * (1.0 - BOX_SHRINK) * s);
        for (uint32_t z = 0; z < grid.dims.z; ++z){
            if (slice >= 0 && z != static_cast<uint32_t>(slice)) continue;
            for (uint32_t y = 0; y < grid.dims.y; ++y){
                for (uint32_t x = 0; x < grid.dims.x; ++x){
                    if (grid.at(x, y, z) != layer.flags) continue;
                    const Eigen::Vector3d lo = toEigen(grid.origin + glm::vec3(x, y, z) * grid.voxelSize);
                    appendBox(layer.lines, lo + inset, lo + Eigen::Vector3d::Constant(s) - inset, layer.color);
                }
            }
        }
    }

    void upload(const Layer& layer){
        auto& scene3d = *scene->GetScene();
        if (scene3d.HasGeometry(layer.name)) scene3d.RemoveGeometry(layer.name);
        if (layer.lines.lines_.empty()) return;
        scene3d.AddGeometry(layer.name, &layer.lines, material, false);
        scene3d.ShowGeometry(layer.name, layer.visible);
    }

    void setVisible(Layer& layer, bool visible){
        layer.visible = visible;
        layer.checkbox->SetChecked(visible);
        if (scene->GetScene()->HasGeometry(layer.name)) scene->GetScene()->ShowGeometry(layer.name, visible);
        PostRedraw();
    }

    void setSlice(int newSlice){
        slice = std::clamp(newSlice, -1, static_cast<int>(grid.dims.z) - 1);
        singleSlice->SetChecked(slice >= 0);
        sliceSlider->SetEnabled(slice >= 0);
        if (slice >= 0) sliceSlider->SetValue(slice);
        for (size_t i = 0; i < VOXEL_LAYERS; ++i){
            rebuildVoxels(layers[i]);
            upload(layers[i]);
            counts[i]->SetText(std::to_string(layers[i].lines.lines_.size() / 12).c_str());
        }
        PostRedraw();
    }

    std::shared_ptr<gui::Widget> buildPanel(){
        const int em = GetTheme().font_size;
        panel = std::make_shared<gui::Vert>(em / 2, gui::Margins(em));

        auto title = std::make_shared<gui::Label>("Key");
        title->SetTextColor(gui::Color(1.0f, 1.0f, 1.0f));
        panel->AddChild(title);

        for (size_t i = 0; i < layers.size(); ++i){
            Layer& layer = layers[i];
            auto row = std::make_shared<gui::Horiz>(em / 2);

            auto swatch = std::make_shared<gui::Label>("    ");
            swatch->SetBackgroundColor(gui::Color(layer.color.x(), layer.color.y(), layer.color.z()));
            row->AddChild(swatch);

            layer.checkbox = std::make_shared<gui::Checkbox>(layer.name.c_str());
            layer.checkbox->SetChecked(layer.visible);
            layer.checkbox->SetOnChecked([this, &layer](bool checked){ setVisible(layer, checked); });
            row->AddChild(layer.checkbox);
            row->AddStretch();

            if (i < VOXEL_LAYERS){
                counts[i] = std::make_shared<gui::Label>(std::to_string(layer.lines.lines_.size() / 12).c_str());
                row->AddChild(counts[i]);
            }
            panel->AddChild(row);
        }
        auto valueHint = std::make_shared<gui::Label>("Voxel values: 1 inside, 2 surface, 3 both.\nCounts are voxels currently drawn.");
        valueHint->SetTextColor(gui::Color(0.7f, 0.7f, 0.7f));
        panel->AddChild(valueHint);
        panel->AddFixed(em);

        singleSlice = std::make_shared<gui::Checkbox>("single Z slice");
        singleSlice->SetOnChecked([this](bool checked){ setSlice(checked ? static_cast<int>(grid.dims.z) / 2 : -1); });
        panel->AddChild(singleSlice);
        sliceSlider = std::make_shared<gui::Slider>(gui::Slider::INT);
        sliceSlider->SetLimits(0, grid.dims.z - 1);
        sliceSlider->SetEnabled(false);
        sliceSlider->SetOnValueChanged([this](double value){ setSlice(static_cast<int>(value)); });
        panel->AddChild(sliceSlider);
        panel->AddFixed(em);

        const std::string info = "Grid " + std::to_string(grid.dims.x) + " x " + std::to_string(grid.dims.y) + " x " +
                                 std::to_string(grid.dims.z) + ", voxel size " + std::to_string(grid.voxelSize) + "\n" +
                                 std::to_string(mesh.indices.size() / 3) + " triangles";
        panel->AddChild(std::make_shared<gui::Label>(info.c_str()));
        panel->AddFixed(em);

        auto keys = std::make_shared<gui::Label>(
            "Keys\n"
            "1 / 2 / 3   toggle voxel layers\n"
            "M           toggle mesh edges\n"
            "Up / Down   step Z slice\n"
            "A           all slices\n"
            "Mouse: drag rotate, scroll zoom,\n"
            "  shift-drag or right-drag pan");
        keys->SetTextColor(gui::Color(0.7f, 0.7f, 0.7f));
        panel->AddChild(keys);
        panel->AddStretch();
        return panel;
    }

    bool onKey(const gui::KeyEvent& e){
        if (e.type != gui::KeyEvent::DOWN) return false;
        if (e.key >= gui::KEY_1 && e.key < gui::KEY_1 + VOXEL_LAYERS){
            Layer& layer = layers[e.key - gui::KEY_1];
            setVisible(layer, !layer.visible);
            return true;
        }
        switch (e.key){
            case gui::KEY_M: setVisible(layers[MESH_LAYER], !layers[MESH_LAYER].visible); return true;
            case gui::KEY_UP: setSlice(slice < 0 ? static_cast<int>(grid.dims.z) / 2 : slice + 1); return true;
            case gui::KEY_DOWN: setSlice(slice < 0 ? static_cast<int>(grid.dims.z) / 2 : std::max(slice - 1, 0)); return true;
            case gui::KEY_A: setSlice(-1); return true;
            default: return false;
        }
    }

    Mesh mesh;
    VoxelGrid grid;
    int slice = -1;
    std::array<Layer, 4> layers;
    std::array<std::shared_ptr<gui::Label>, VOXEL_LAYERS> counts;
    rendering::MaterialRecord material;
    std::shared_ptr<gui::SceneWidget> scene;
    std::shared_ptr<gui::Vert> panel;
    std::shared_ptr<gui::Checkbox> singleSlice;
    std::shared_ptr<gui::Slider> sliceSlider;
};

}  // namespace

int main(int argc, char** argv){
    const std::string objPath = argc > 1 ? argv[1] : STEAK_OBJ;
    const float voxelSize = argc > 2 ? std::strtof(argv[2], nullptr) : 0.5f;

    try{
        Mesh mesh = loadObj(objPath);
        VoxelGrid grid;
        {
            ComputeContext ctx("voxel_viewer");
            Voxelizer voxelizer(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, VOXELIZE_SPV);
            grid = voxelizer.voxelize(mesh.positions, mesh.indices, voxelSize);
        }
        std::cout << "Voxelized " << mesh.indices.size() / 3 << " triangles into " << grid.dims.x << "x" << grid.dims.y
                  << "x" << grid.dims.z << " grid (voxel size " << grid.voxelSize << ")" << std::endl;

        auto& app = gui::Application::GetInstance();
        app.Initialize(OPEN3D_RESOURCE_DIR);
        app.AddWindow(std::make_shared<ViewerWindow>("voxel viewer: " + objPath, std::move(mesh), std::move(grid)));
        app.Run();
    } catch (const std::exception& e){
        std::cerr << "voxel_viewer: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
