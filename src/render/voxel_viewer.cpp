// Debug viewer: voxelizes an .obj on the GPU and draws every non-empty voxel as a box outline.
// The side panel is the color key, a reset button, and collapsible settings.
// Typing in a box stores that value; the voxels rebuild when a reset finishes.
//
// Usage: voxel_viewer [mesh.obj|mesh.gltf|mesh.glb] [voxelSize]
//
// Keys:
//   1        toggle inside-only voxels      (flags == VOXEL_INSIDE)
//   2        toggle surface-only voxels     (flags == VOXEL_SURFACE)
//   3        toggle inside + surface voxels (flags == both)
//   M        toggle the mesh's triangle edges
//   X / Y / Z  choose the slice axis
//   Up/Down    show a single slice and move it
//   A          show all slices again

#include "compute_context.hpp"
#include "mesh_loader.hpp"
#include "voxelizer.hpp"

#include "reset_dialog.hpp"

#include <open3d/Open3D.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace gui = open3d::visualization::gui;
namespace rendering = open3d::visualization::rendering;

namespace {

constexpr double BOX_SHRINK = 0.85;
constexpr double MAX_CELLS_PER_SIDE = 500.0;
constexpr double MM_PER_MODEL_UNIT = 10.0;  // steak.obj is centimeters; the settings field is millimeters
constexpr std::array<const char*, 3> AXIS_NAMES{"X", "Y", "Z"};

Eigen::Vector3d toEigen(glm::vec3 v){ return {v.x, v.y, v.z}; }

void appendBox(open3d::geometry::LineSet& lines, Eigen::Vector3d lo, Eigen::Vector3d hi, const Eigen::Vector3d& color){
    static constexpr std::array<std::array<int, 2>, 12> edges{{
        {0, 1}, {2, 3}, {4, 5}, {6, 7},
        {0, 2}, {1, 3}, {4, 6}, {5, 7},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
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

std::shared_ptr<gui::ImageWidget> makeSwatch(const Eigen::Vector3d& color, int width, int height){
    auto image = std::make_shared<open3d::geometry::Image>();
    image->Prepare(width, height, 3, 1);
    for (int i = 0; i < width * height; ++i){
        for (int c = 0; c < 3; ++c) image->data_[3 * i + c] = static_cast<uint8_t>(std::lround(255.0 * color[c]));
    }
    return std::make_shared<gui::ImageWidget>(image);
}

std::string format(const char* fmt, double value){
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), fmt, value);
    return buffer;
}

bool parseNumber(const char* text, double& value, std::string& error){
    if (!text){
        error = "Enter a number.";
        return false;
    }
    while (*text == ' ' || *text == '\t') ++text;
    if (*text == '\0'){
        error = "Enter a number.";
        return false;
    }
    char* end = nullptr;
    value = std::strtod(text, &end);
    if (end == text){
        error = "Enter a number.";
        return false;
    }
    while (*end == ' ' || *end == '\t') ++end;
    if (*end != '\0' || !std::isfinite(value)){
        error = "Enter a number.";
        return false;
    }
    return true;
}

float meshExtent(const Mesh& mesh){
    if (mesh.positions.empty()) return 0.0f;
    glm::vec3 low(std::numeric_limits<float>::infinity());
    glm::vec3 high(-std::numeric_limits<float>::infinity());
    for (const glm::vec3& point : mesh.positions){
        low = glm::min(low, point);
        high = glm::max(high, point);
    }
    return std::max({high.x - low.x, high.y - low.y, high.z - low.z});
}

struct VoxelDraw{
    Eigen::Vector3d color;
    uint32_t flags = 0;
};

bool onSlice(int sliceAxis, int slice, uint32_t x, uint32_t y, uint32_t z){
    if (slice < 0) return true;
    const uint32_t coord = sliceAxis == 0 ? x : sliceAxis == 1 ? y : z;
    return coord == static_cast<uint32_t>(slice);
}

std::array<open3d::geometry::LineSet, 3> buildVoxelLines(const VoxelGrid& grid, int sliceAxis, int slice,
                                                        const std::array<VoxelDraw, 3>& draws,
                                                        const std::function<void(float)>& progress){
    std::array<open3d::geometry::LineSet, 3> lines;
    const double s = grid.voxelSize;
    const Eigen::Vector3d inset = Eigen::Vector3d::Constant(0.5 * (1.0 - BOX_SHRINK) * s);
    const uint32_t zCount = std::max(1u, grid.dims.z);
    for (uint32_t layer = 0; layer < draws.size(); ++layer){
        for (uint32_t z = 0; z < grid.dims.z; ++z){
            for (uint32_t y = 0; y < grid.dims.y; ++y){
                for (uint32_t x = 0; x < grid.dims.x; ++x){
                    if (!onSlice(sliceAxis, slice, x, y, z) || grid.at(x, y, z) != draws[layer].flags) continue;
                    const Eigen::Vector3d lo = toEigen(grid.origin + glm::vec3(x, y, z) * grid.voxelSize);
                    appendBox(lines[layer], lo + inset, lo + Eigen::Vector3d::Constant(s) - inset, draws[layer].color);
                }
            }
            if (progress) progress((layer * zCount + z + 1.0f) / (draws.size() * zCount));
        }
    }
    if (progress && grid.dims.z == 0) progress(1.0f);
    return lines;
}

struct Layer{
    std::string name;
    Eigen::Vector3d color;
    uint32_t flags = 0;
    bool visible = true;
    open3d::geometry::LineSet lines;
    std::shared_ptr<gui::Checkbox> checkbox;
};

class ViewerWindow : public gui::Window{
public:
    ViewerWindow(const std::string& title, Mesh meshIn, VoxelGrid gridIn, gui::FontId headingFontIn)
        : gui::Window(title, 1400, 850), mesh(std::move(meshIn)), grid(std::move(gridIn)), editedVoxelSize(grid.voxelSize),
          headingFont(headingFontIn){
        layers = {{
            {"inside only", {0.20, 0.45, 0.95}, VOXEL_INSIDE},
            {"surface only", {1.00, 0.60, 0.10}, VOXEL_SURFACE},
            {"inside + surface", {0.90, 0.15, 0.15}, VOXEL_INSIDE | VOXEL_SURFACE},
            {"mesh edges", {0.55, 0.55, 0.55}},
        }};
        buildMeshEdges();
        const_cast<gui::Theme&>(GetTheme()).text_edit_background_color = gui::Color(0.42f, 0.44f, 0.48f);

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
        SetOnClose([this](){ return !resetting; });
        SetOnTickEvent([this](){ return onTick(); });
    }

    void Layout(const gui::LayoutContext& context) override{
        const gui::Rect r = GetContentRect();
        const int panelWidth = std::min(r.width / 2, 34 * context.theme.font_size);
        panel->SetFrame(gui::Rect(r.GetRight() - panelWidth, r.y, panelWidth, r.height));
        scene->SetFrame(gui::Rect(r.x, r.y, r.width - panelWidth, r.height));
        gui::Window::Layout(context);
    }

private:
    static constexpr size_t VOXEL_LAYERS = 3;
    static constexpr size_t MESH_LAYER = 3;

    struct SettingField{
        std::function<std::string()> current;
        std::function<bool(const std::string&, std::string&)> commit;
        std::shared_ptr<gui::TextEdit> edit;
        std::shared_ptr<gui::Label> value;
    };

    void buildMeshEdges(){
        auto& lines = layers[MESH_LAYER].lines;
        for (const glm::vec3& p : mesh.positions) lines.points_.push_back(toEigen(p));
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3){
            for (int e = 0; e < 3; ++e) lines.lines_.emplace_back(mesh.indices[t + e], mesh.indices[t + (e + 1) % 3]);
        }
        lines.PaintUniformColor(layers[MESH_LAYER].color);
    }

    void rebuildVoxels(Layer& layer){
        layer.lines.Clear();
        const double s = grid.voxelSize;
        const Eigen::Vector3d inset = Eigen::Vector3d::Constant(0.5 * (1.0 - BOX_SHRINK) * s);
        for (uint32_t z = 0; z < grid.dims.z; ++z){
            for (uint32_t y = 0; y < grid.dims.y; ++y){
                for (uint32_t x = 0; x < grid.dims.x; ++x){
                    if (!onSlice(sliceAxis, slice, x, y, z) || grid.at(x, y, z) != layer.flags) continue;
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

    void syncSliceControls(){
        const bool on = slice >= 0;
        singleSlice->SetChecked(on);
        sliceSlider->SetEnabled(on);
        axisCombo->SetEnabled(on);
        axisLabel->SetTextColor(on ? gui::Color(0.9f, 0.9f, 0.9f) : gui::Color(0.45f, 0.45f, 0.45f));
        if (on) sliceSlider->SetValue(slice);
    }

    void setSlice(int newSlice){
        if (resetting) return;
        const int limit = static_cast<int>(grid.dims[sliceAxis]) - 1;
        slice = std::clamp(newSlice, -1, std::max(limit, -1));
        syncSliceControls();
        for (size_t i = 0; i < VOXEL_LAYERS; ++i){
            rebuildVoxels(layers[i]);
            upload(layers[i]);
            counts[i]->SetText(std::to_string(layers[i].lines.lines_.size() / 12).c_str());
        }
        PostRedraw();
    }

    void setSliceAxis(int axis){
        if (resetting) return;
        sliceAxis = axis;
        axisCombo->SetSelectedIndex(axis);
        sliceSlider->SetLimits(0, std::max(1u, grid.dims[axis]) - 1u);
        setSlice(slice < 0 ? -1 : static_cast<int>(grid.dims[axis]) / 2);
    }

    void refreshFields(){
        for (SettingField& field : fields){
            const std::string text = field.current();
            if (field.edit) field.edit->SetText(text.c_str());
            if (field.value) field.value->SetText(text.c_str());
        }
    }

    void onFieldChanged(size_t index, const char* text){
        std::string error;
        if (!fields[index].commit(text, error)) ShowMessageBox("Invalid value", error.c_str());
        refreshFields();
    }

    bool voxelSizeOk(double size, std::string& error) const{
        if (!(size > 0.0)){
            error = "Voxel size must be positive.";
            return false;
        }
        const float extent = meshExtent(mesh);
        if (extent > 0.0f && extent / static_cast<float>(size) > MAX_CELLS_PER_SIDE){
            error = "Voxel size is too small; the grid would exceed 512 cells on a side.";
            return false;
        }
        return true;
    }

    void addField(gui::Widget& parent, std::string caption, bool editable, std::function<std::string()> current,
                  std::function<bool(const std::string&, std::string&)> commit){
        const size_t index = fields.size();
        SettingField field;
        field.current = std::move(current);
        field.commit = std::move(commit);

        auto block = std::make_shared<gui::Vert>(2);
        auto label = std::make_shared<gui::Label>(caption.c_str());
        label->SetTextColor(gui::Color(0.82f, 0.82f, 0.82f));
        block->AddChild(label);
        if (editable){
            field.edit = std::make_shared<gui::TextEdit>();
            field.edit->SetText(field.current().c_str());
            field.edit->SetOnValueChanged([this, index](const char* text){ onFieldChanged(index, text); });
            block->AddChild(field.edit);
        } else{
            field.value = std::make_shared<gui::Label>(field.current().c_str());
            block->AddChild(field.value);
        }
        parent.AddChild(block);
        fields.push_back(std::move(field));
    }

    std::shared_ptr<gui::Widget> buildPanel(){
        const int em = GetTheme().font_size;
        panel = std::make_shared<gui::Vert>(em / 2, gui::Margins(em));
        fields.reserve(8);

        auto title = std::make_shared<gui::Label>("Key");
        title->SetTextColor(gui::Color(1.0f, 1.0f, 1.0f));
        panel->AddChild(title);

        for (size_t i = 0; i < layers.size(); ++i){
            Layer& layer = layers[i];
            auto row = std::make_shared<gui::Horiz>(em / 2);
            row->AddChild(std::make_shared<steakui::Chrome>(makeSwatch(layer.color, 2 * em, em)));
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
        //auto valueHint = std::make_shared<gui::Label>("Voxel values: 1 inside, 2 surface, 3 both.\nCounts are voxels currently drawn.");
        //valueHint->SetTextColor(gui::Color(0.7f, 0.7f, 0.7f));
        //panel->AddChild(valueHint);
        panel->AddFixed(em / 2);

        auto resetButton = std::make_shared<gui::Button>("Reset voxels");
        resetButton->SetOnClicked([this](){ askReset(); });
        panel->AddChild(resetButton);
        panel->AddFixed(em / 2);

        auto settings = std::make_shared<gui::Label>("Settings");
        settings->SetFontId(headingFont);
        settings->SetTextColor(gui::Color(1.0f, 1.0f, 1.0f));
        panel->AddChild(settings);
        auto hint = std::make_shared<gui::Label>("Type a value, then reset to rebuild the voxels.");
        hint->SetTextColor(gui::Color(0.7f, 0.7f, 0.7f));
        panel->AddChild(hint);

        auto scroll = std::make_shared<gui::ScrollableVert>(em / 2, gui::Margins(0, em / 3, em + em / 2, 0));
        auto relayout = [this](){
            Layout(GetLayoutContext());
            PostRedraw();
        };
        auto view = std::make_shared<steakui::Disclosure>("View", headingFont, em);
        view->SetOnToggled(relayout);
        singleSlice = std::make_shared<gui::Checkbox>("Single slice");
        singleSlice->SetOnChecked([this](bool checked){ setSlice(checked ? static_cast<int>(grid.dims[sliceAxis]) / 2 : -1); });
        view->content().AddChild(singleSlice);
        sliceSlider = std::make_shared<steakui::TrackSlider>();
        sliceSlider->SetLimits(0, std::max(1u, grid.dims[sliceAxis]) - 1u);
        sliceSlider->SetEnabled(false);
        sliceSlider->SetOnValueChanged([this](double value){ setSlice(static_cast<int>(value)); });
        view->content().AddChild(sliceSlider);
        axisLabel = std::make_shared<gui::Label>("Slice Axis");
        axisLabel->SetTextColor(gui::Color(0.45f, 0.45f, 0.45f));
        axisCombo = std::make_shared<gui::Combobox>(std::vector<const char*>(AXIS_NAMES.begin(), AXIS_NAMES.end()));
        axisCombo->SetSelectedIndex(sliceAxis);
        axisCombo->SetEnabled(false);
        axisCombo->SetOnValueChanged([this](const char*, int index){ setSliceAxis(index); });
        view->content().AddChild(std::make_shared<steakui::SliceAxisRow>(axisLabel, axisCombo));
        scroll->AddChild(view);

        auto gridSection = std::make_shared<steakui::Disclosure>("Grid", headingFont, em);
        gridSection->SetOnToggled(relayout);
        addField(gridSection->content(), "Voxel size (mm)", true, [this](){ return format("%.6g", editedVoxelSize * MM_PER_MODEL_UNIT); },
                 [this](const std::string& text, std::string& error){
                     double mm = 0.0;
                     if (!parseNumber(text.c_str(), mm, error)) return false;
                     if (!(mm > 0.0)){
                         error = "Voxel size must be positive.";
                         return false;
                     }
                     const double modelUnits = mm / MM_PER_MODEL_UNIT;
                     if (!voxelSizeOk(modelUnits, error)) return false;
                     editedVoxelSize = static_cast<float>(modelUnits);
                     return true;
                 });
        const char* dimNames[] = {"Cells X", "Cells Y", "Cells Z"};
        for (int axis = 0; axis < 3; ++axis){
            addField(gridSection->content(), dimNames[axis], false, [this, axis](){ return std::to_string(grid.dims[axis]); }, {});
        }
        addField(gridSection->content(), "Triangles", false, [this](){ return std::to_string(mesh.indices.size() / 3); }, {});
        scroll->AddChild(gridSection);

        auto keys = std::make_shared<gui::Label>(
            "Drag mouse to rotate, scroll to zoom, shift-drag to pan\n"
            "Press 1, 2, 3 to view inside, surface, or both voxel types\n"
            "Press 'M' to toggle original object mesh\n"
            "Press 'X', 'Y', 'Z' to take slices along axes and move them with the arrow keys"
            "A           all slices\n"
        );
        keys->SetTextColor(gui::Color(0.7f, 0.7f, 0.7f));
        scroll->AddChild(keys);
        panel->AddChild(scroll);
        return panel;
    }

    void askReset(){
        if (resetting) return;
        std::string error;
        if (!voxelSizeOk(editedVoxelSize, error)){
            ShowMessageBox("Cannot reset", error.c_str());
            return;
        }
        const int width = std::max(320, 28 * GetTheme().font_size);
        auto dialog = std::make_shared<steakui::ConfirmDialog>(
            "Reset voxels", "Reset the voxels with the saved settings?", width, [this](bool accepted){
                gui::Application::GetInstance().PostToMainThread(this, [this, accepted](){
                    steakui::chromeHidden() = false;
                    CloseDialog();
                    if (accepted) startReset();
                });
            });
        steakui::chromeHidden() = true;
        ShowDialog(dialog);
    }

    void startReset(){
        if (resetting) return;
        resetting = true;
        const int width = std::max(320, 28 * GetTheme().font_size);
        progressDialog = std::make_shared<steakui::ResetProgressDialog>("Rebuilding the voxels.", width);
        progressDialog->SetFraction(0.0f);
        steakui::chromeHidden() = true;
        ShowDialog(progressDialog);

        const float voxelSize = editedVoxelSize;
        const int axis = sliceAxis;
        const int sliceNow = slice;
        std::array<VoxelDraw, 3> draws;
        for (size_t i = 0; i < VOXEL_LAYERS; ++i) draws[i] = {layers[i].color, layers[i].flags};
        const std::vector<glm::vec3> positions = mesh.positions;
        const std::vector<uint32_t> indices = mesh.indices;
        auto dialog = progressDialog;
        gui::Application::GetInstance().RunInThread([this, dialog, voxelSize, axis, sliceNow, draws, positions, indices](){
            int lastPercent = -1;
            auto report = [this, dialog, &lastPercent](float fraction){
                const int percent = std::clamp(static_cast<int>(std::lround(fraction * 100.0f)), 0, 100);
                if (percent == lastPercent) return;
                lastPercent = percent;
                gui::Application::GetInstance().PostToMainThread(this, [dialog, percent](){ dialog->SetFraction(percent / 100.0f); });
            };
            try{
                report(0.01f);
                VoxelGrid next;
                {
                    ComputeContext ctx("voxel_viewer");
                    report(0.08f);
                    Voxelizer voxelizer(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, VOXELIZE_SPV);
                    next = voxelizer.voxelize(positions, indices, voxelSize);
                }
                report(0.55f);
                const int shown = sliceNow < 0 ? -1 : std::min(sliceNow, std::max(0, static_cast<int>(next.dims[axis]) - 1));
                auto lines = std::make_shared<std::array<open3d::geometry::LineSet, 3>>(
                    buildVoxelLines(next, axis, shown, draws, [&](float fraction){ report(0.55f + 0.44f * fraction); }));
                auto gridPtr = std::make_shared<VoxelGrid>(std::move(next));
                gui::Application::GetInstance().PostToMainThread(
                    this, [this, gridPtr, lines, axis, shown](){ finishReset(std::move(*gridPtr), std::move(*lines), axis, shown); });
            } catch (const std::exception& e){
                const std::string message = e.what();
                gui::Application::GetInstance().PostToMainThread(this, [this, message](){
                    steakui::chromeHidden() = false;
                    CloseDialog();
                    progressDialog.reset();
                    resetting = false;
                    ShowMessageBox("Reset failed", message.c_str());
                });
            }
        });
    }

    void finishReset(VoxelGrid next, std::array<open3d::geometry::LineSet, 3> lines, int axis, int shown){
        const bool shapeChanged = next.dims != grid.dims || next.voxelSize != grid.voxelSize;
        grid = std::move(next);
        editedVoxelSize = grid.voxelSize;
        sliceAxis = axis;
        slice = shown;
        for (size_t i = 0; i < VOXEL_LAYERS; ++i){
            layers[i].lines = std::move(lines[i]);
            upload(layers[i]);
            counts[i]->SetText(std::to_string(layers[i].lines.lines_.size() / 12).c_str());
        }
        if (shapeChanged){
            const auto& bounds = scene->GetScene()->GetBoundingBox();
            scene->SetupCamera(60.0f, bounds, bounds.GetCenter().cast<float>());
        }
        axisCombo->SetSelectedIndex(sliceAxis);
        sliceSlider->SetLimits(0, std::max(1u, grid.dims[sliceAxis]) - 1u);
        syncSliceControls();
        refreshFields();
        PostRedraw();
        progressDialog->SetFraction(1.0f);
        progressDoneAt = gui::Application::GetInstance().Now() + 0.25;
        progressArmed = true;
    }

    bool onTick(){
        if (!progressArmed) return false;
        if (gui::Application::GetInstance().Now() < progressDoneAt) return true;
        steakui::chromeHidden() = false;
        CloseDialog();
        progressDialog.reset();
        progressArmed = false;
        resetting = false;
        return true;
    }

    bool onKey(const gui::KeyEvent& e){
        if (e.type != gui::KeyEvent::DOWN) return false;
        if (resetting) return true;
        if (e.key >= gui::KEY_1 && e.key < gui::KEY_1 + static_cast<int>(VOXEL_LAYERS)){
            Layer& layer = layers[static_cast<size_t>(e.key - gui::KEY_1)];
            setVisible(layer, !layer.visible);
            return true;
        }
        switch (e.key){
            case gui::KEY_M: setVisible(layers[MESH_LAYER], !layers[MESH_LAYER].visible); return true;
            case gui::KEY_X: setSliceAxis(0); return true;
            case gui::KEY_Y: setSliceAxis(1); return true;
            case gui::KEY_Z: setSliceAxis(2); return true;
            case gui::KEY_UP: setSlice(slice < 0 ? static_cast<int>(grid.dims[sliceAxis]) / 2 : slice + 1); return true;
            case gui::KEY_DOWN: setSlice(slice < 0 ? static_cast<int>(grid.dims[sliceAxis]) / 2 : std::max(slice - 1, 0)); return true;
            case gui::KEY_A: setSlice(-1); return true;
            default: return false;
        }
    }

    Mesh mesh;
    VoxelGrid grid;
    float editedVoxelSize = 0.0f;
    gui::FontId headingFont = 0;
    int sliceAxis = 2;
    int slice = -1;
    bool resetting = false;
    bool progressArmed = false;
    double progressDoneAt = 0.0;
    std::array<Layer, 4> layers;
    std::array<std::shared_ptr<gui::Label>, VOXEL_LAYERS> counts;
    rendering::MaterialRecord material;
    std::shared_ptr<gui::SceneWidget> scene;
    std::shared_ptr<gui::Vert> panel;
    std::shared_ptr<gui::Checkbox> singleSlice;
    std::shared_ptr<steakui::TrackSlider> sliceSlider;
    std::shared_ptr<gui::Label> axisLabel;
    std::shared_ptr<gui::Combobox> axisCombo;
    std::shared_ptr<steakui::ResetProgressDialog> progressDialog;
    std::vector<SettingField> fields;
};

gui::FontId headingFont(){
    try{
        return gui::Application::GetInstance().AddFont(gui::FontDescription(gui::FontDescription::SANS_SERIF, gui::FontStyle::BOLD, 20));
    } catch (const std::exception& e){
        std::cerr << "voxel_viewer: heading font unavailable (" << e.what() << ")\n";
        return gui::Application::DEFAULT_FONT_ID;
    }
}

}  // namespace

int main(int argc, char** argv){
    const std::string objPath = argc > 1 ? argv[1] : STEAK_OBJ;
    float voxelSize = argc > 2 ? std::strtof(argv[2], nullptr) : 0.5f;

    try{
        Mesh mesh = dropOverlappingParts(loadMesh(objPath));
        if (argc <= 2 && !mesh.positions.empty()){
            glm::vec3 low(std::numeric_limits<float>::infinity());
            glm::vec3 high(-std::numeric_limits<float>::infinity());
            for (const glm::vec3& point : mesh.positions){
                low = glm::min(low, point);
                high = glm::max(high, point);
            }
            const float extent = std::max({high.x - low.x, high.y - low.y, high.z - low.z});
            if (voxelSize > 0.0f && extent / voxelSize > 200.0f) voxelSize = extent / 80.0f;
        }
        VoxelGrid grid;
        {
            ComputeContext ctx("voxel_viewer");
            Voxelizer voxelizer(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, VOXELIZE_SPV);
            grid = voxelizer.voxelize(mesh.positions, mesh.indices, voxelSize);
        }
        std::cout << "Voxelized " << mesh.indices.size() / 3 << " triangles into " << grid.dims.x << "x" << grid.dims.y << "x"
                  << grid.dims.z << " grid (voxel size " << grid.voxelSize << ")" << std::endl;

        auto& app = gui::Application::GetInstance();
        app.Initialize(OPEN3D_RESOURCE_DIR);
        const gui::FontId font = headingFont();
        app.AddWindow(std::make_shared<ViewerWindow>("voxel viewer: " + objPath, std::move(mesh), std::move(grid), font));
        app.Run();
    } catch (const std::exception& e){
        std::cerr << "voxel_viewer: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
