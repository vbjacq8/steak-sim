// Heat-map window shared by heat_viewer and steak_heat_demo. Only the faces between a drawn cell and
// an undrawn one (air, hidden pan, outside the slice, or outside the grid) are built, so a 128^3 grid
// stays at ~100k quads.
//
// The side panel keeps a temperature key, then a reset button, then collapsible settings. Typing in a
// box stores that value; the blocks change when a reset finishes.

#include "heat_view.hpp"

#include "reset_dialog.hpp"

#include <open3d/Open3D.h>
#include <open3d/visualization/rendering/ColorGrading.h>
#include <open3d/visualization/rendering/View.h>

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

constexpr double BLOCK_SHRINK = 0.9;
constexpr double MM_PER_M = 1000.0;
constexpr int LEGEND_ROWS = 7;
constexpr float COLOR_MIN_F = 70.0f;
constexpr float COLOR_MAX_F = 200.0f;
constexpr int MAX_DIM = 256;
constexpr long MAX_STEPS = 100000;
constexpr std::array<const char*, 3> AXIS_NAMES{"X", "Y", "Z"};

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

bool parseCount(const char* text, long& value, long min, long max, std::string& error, const char* what){
    double number = 0.0;
    if (!parseNumber(text, number, error)) return false;
    if (number < min || number > max || std::abs(number - std::round(number)) > 1e-6){
        error = std::string(what) + " must be a whole number from " + std::to_string(min) + " to " + std::to_string(max) + ".";
        return false;
    }
    value = std::lround(number);
    return true;
}

Eigen::Vector3d heatColor(double t){
    static constexpr std::array<std::array<double, 3>, 5> stops{{
        {0.12, 0.25, 0.85}, {0.10, 0.75, 0.90}, {0.98, 0.88, 0.15}, {0.98, 0.45, 0.05}, {0.75, 0.05, 0.05}}};
    const double x = std::clamp(t, 0.0, 1.0) * (stops.size() - 1);
    const size_t i = std::min(static_cast<size_t>(x), stops.size() - 2);
    const double f = x - i;
    return {stops[i][0] + f * (stops[i + 1][0] - stops[i][0]), stops[i][1] + f * (stops[i + 1][1] - stops[i][1]),
            stops[i][2] + f * (stops[i + 1][2] - stops[i][2])};
}

std::shared_ptr<gui::ImageWidget> makeSwatch(const Eigen::Vector3d& color, int width, int height){
    auto image = std::make_shared<open3d::geometry::Image>();
    image->Prepare(width, height, 3, 1);
    for (int i = 0; i < width * height; ++i){
        for (int c = 0; c < 3; ++c) image->data_[3 * i + c] = static_cast<uint8_t>(std::lround(255.0 * color[c]));
    }
    return std::make_shared<gui::ImageWidget>(image);
}

void appendFace(open3d::geometry::TriangleMesh& blocks, const HeatGrid& grid, glm::uvec3 cell, int axis, bool positive,
                float coldTemp, float hotTemp){
    const int u = (axis + 1) % 3;
    const int v = (axis + 2) % 3;
    const double s = grid.spacing * MM_PER_M;
    const double inset = 0.5 * (1.0 - BLOCK_SHRINK) * s;
    Eigen::Vector3d lo, hi;
    for (int a = 0; a < 3; ++a){
        lo[a] = cell[a] * s + inset;
        hi[a] = (cell[a] + 1) * s - inset;
    }
    const double plane = positive ? hi[axis] : lo[axis];
    const std::array<std::array<double, 2>, 4> corners{{{lo[u], lo[v]}, {hi[u], lo[v]}, {hi[u], hi[v]}, {lo[u], hi[v]}}};

    const int base = static_cast<int>(blocks.vertices_.size());
    const double t = hotTemp > coldTemp ? (grid.at(cell.x, cell.y, cell.z) - coldTemp) / (hotTemp - coldTemp) : 1.0;
    const Eigen::Vector3d color = heatColor(t).array().pow(2.2);
    for (const auto& [cu, cv] : corners){
        Eigen::Vector3d p;
        p[axis] = plane;
        p[u] = cu;
        p[v] = cv;
        blocks.vertices_.push_back(p);
        blocks.vertex_colors_.push_back(color);
    }
    if (positive){
        blocks.triangles_.emplace_back(base, base + 1, base + 2);
        blocks.triangles_.emplace_back(base, base + 2, base + 3);
    } else{
        blocks.triangles_.emplace_back(base, base + 2, base + 1);
        blocks.triangles_.emplace_back(base, base + 3, base + 2);
    }
}

// Builds the block mesh for the current slice. Safe to call off the UI thread. `progress` is the fraction of
// z-slices finished.
open3d::geometry::TriangleMesh buildHeatMesh(const HeatGrid& grid, int sliceAxis, int slice, bool showPan, float coldTemp,
                                             float hotTemp, const std::function<void(float)>& progress){
    glm::uvec3 regionLo(0);
    glm::uvec3 regionHi = grid.dims - 1u;
    if (slice >= 0){
        const auto clamped = static_cast<uint32_t>(std::min(slice, static_cast<int>(grid.dims[sliceAxis]) - 1));
        regionLo[sliceAxis] = regionHi[sliceAxis] = clamped;
    }

    auto drawn = [&](glm::ivec3 cell){
        if (!glm::all(glm::greaterThanEqual(cell, glm::ivec3(regionLo))) || !glm::all(glm::lessThanEqual(cell, glm::ivec3(regionHi))))
            return false;
        const uint32_t kind = grid.kind(cell.x, cell.y, cell.z);
        return kind == CELL_SOLID || (kind == CELL_PAN && showPan);
    };

    open3d::geometry::TriangleMesh blocks;
    const uint32_t zSpan = regionHi.z - regionLo.z + 1u;
    uint32_t zDone = 0;
    for (uint32_t z = regionLo.z; z <= regionHi.z; ++z){
        for (uint32_t y = regionLo.y; y <= regionHi.y; ++y){
            for (uint32_t x = regionLo.x; x <= regionHi.x; ++x){
                const glm::ivec3 cell(x, y, z);
                if (!drawn(cell)) continue;
                for (int axis = 0; axis < 3; ++axis){
                    for (int side : {-1, 1}){
                        glm::ivec3 neighbor = cell;
                        neighbor[axis] += side;
                        if (!drawn(neighbor)) appendFace(blocks, grid, glm::uvec3(cell), axis, side > 0, coldTemp, hotTemp);
                    }
                }
            }
        }
        if (progress) progress(static_cast<float>(++zDone) / static_cast<float>(zSpan));
    }
    return blocks;
}

class HeatViewWindow : public gui::Window{
public:
    HeatViewWindow(const std::string& title, HeatGrid gridIn, HeatParams paramsIn, uint32_t stepsIn, HeatSimulator simulateIn,
                   bool gridFollowsMeshIn, gui::FontId headingFontIn)
        : gui::Window(title, 1400, 850),
          grid(std::move(gridIn)),
          edited(paramsIn),
          simulate(std::move(simulateIn)),
          editedSteps(stepsIn),
          gridFollowsMesh(gridFollowsMeshIn),
          headingFont(headingFontIn){
        coldTemp = COLOR_MIN_F;
        hotTemp = COLOR_MAX_F;
        recount();
        // Stock text boxes are nearly black. A lighter fill keeps the number readable.
        const_cast<gui::Theme&>(GetTheme()).text_edit_background_color = gui::Color(0.42f, 0.44f, 0.48f);

        scene = std::make_shared<gui::SceneWidget>();
        scene->SetScene(std::make_shared<rendering::Open3DScene>(GetRenderer()));
        scene->GetScene()->SetBackground({0.08f, 0.08f, 0.10f, 1.0f});
        scene->GetScene()->GetView()->SetColorGrading(rendering::ColorGradingParams(
            rendering::ColorGradingParams::Quality::kHigh, rendering::ColorGradingParams::ToneMapping::kLinear));
        material.shader = "defaultUnlit";
        rebuildBlocks();

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
    struct SettingField{
        std::function<std::string()> current;
        std::function<bool(const std::string&, std::string&)> commit;
        std::shared_ptr<gui::TextEdit> edit;
        std::shared_ptr<gui::Label> value;
    };

    void recount(){
        solidCells = 0;
        panContactCells = 0;
        minTemp = std::numeric_limits<float>::max();
        maxTemp = std::numeric_limits<float>::lowest();
        for (size_t i = 0; i < grid.cellCount(); ++i){
            if (grid.cells[i] != CELL_SOLID) continue;
            ++solidCells;
            minTemp = std::min(minTemp, grid.temperature[i]);
            maxTemp = std::max(maxTemp, grid.temperature[i]);
            const size_t below = i - grid.dims.x;
            if (i % (size_t(grid.dims.x) * grid.dims.y) >= grid.dims.x && grid.cells[below] == CELL_PAN) ++panContactCells;
        }
    }

    void uploadBlocks(){
        auto& scene3d = *scene->GetScene();
        if (scene3d.HasGeometry("blocks")) scene3d.RemoveGeometry("blocks");
        if (!blocks.triangles_.empty()) scene3d.AddGeometry("blocks", &blocks, material, false);
        PostRedraw();
    }

    void rebuildBlocks(){
        if (resetting) return;
        blocks = buildHeatMesh(grid, sliceAxis, slice, showPan, coldTemp, hotTemp, {});
        uploadBlocks();
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
        slice = std::clamp(newSlice, -1, static_cast<int>(grid.dims[sliceAxis]) - 1);
        syncSliceControls();
        rebuildBlocks();
    }

    void setSliceAxis(int axis){
        sliceAxis = axis;
        axisCombo->SetSelectedIndex(axis);
        sliceSlider->SetLimits(0, grid.dims[axis] - 1);
        setSlice(slice < 0 ? -1 : static_cast<int>(grid.dims[axis]) / 2);
    }

    void updateLegend(){
        for (int row = 0; row < LEGEND_ROWS; ++row){
            const double t = 1.0 - static_cast<double>(row) / (LEGEND_ROWS - 1);
            const std::string prefix = row == 0 ? ">= " : row == LEGEND_ROWS - 1 ? "<= " : "";
            legendLabels[row]->SetText((prefix + format("%.0f", coldTemp + t * (hotTemp - coldTemp))).c_str());
        }
    }

    void setColorMax(float value){
        hotTemp = value;
        updateLegend();
        rebuildBlocks();
    }

    void setShowPan(bool show){
        showPan = show;
        panCheckbox->SetChecked(show);
        rebuildBlocks();
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

    // Writes `edited` only after the whole value is valid, so a rejected edit leaves the saved model alone.
    bool commitSteps(long count, std::string& error){
        if (count < 1 || count > MAX_STEPS){
            error = "Steps must be a whole number from 1 to " + std::to_string(MAX_STEPS) + ".";
            return false;
        }
        editedSteps = static_cast<uint32_t>(count);
        edited.duration = static_cast<float>(editedSteps) * edited.timeStep;
        return true;
    }

    bool commitDurationOrStep(double duration, double timeStep, std::string& error){
        if (!(duration > 0.0) || !(timeStep > 0.0)){
            error = "Duration and time step must be positive.";
            return false;
        }
        const double count = std::round(duration / timeStep);
        if (count < 1.0 || count > MAX_STEPS){
            error = "That duration and time step need between 1 and " + std::to_string(MAX_STEPS) + " steps.";
            return false;
        }
        edited.duration = static_cast<float>(duration);
        edited.timeStep = static_cast<float>(timeStep);
        editedSteps = static_cast<uint32_t>(count);
        return true;
    }

    std::string resetError() const{
        if (editedSteps < 1 || editedSteps > MAX_STEPS) return "Steps must be from 1 to " + std::to_string(MAX_STEPS) + ".";
        if (!(edited.spacing > 0.0f) || !(edited.timeStep > 0.0f) || !(edited.diffusivity > 0.0f) || !(edited.conductivity > 0.0f) ||
            edited.panTransfer < 0.0f || edited.airTransfer < 0.0f){
            return "Spacing, time step, diffusivity and conductivity must be positive, and heat transfer cannot be negative.";
        }
        if (!gridFollowsMesh && (edited.dims.x == 0 || edited.dims.y == 0 || edited.dims.z == 0)) return "The grid is empty.";
        if (edited.fourier() > edited.maxStableFourier()){
            const float maxStep = edited.maxStableFourier() * edited.spacing * edited.spacing / edited.diffusivity;
            return "Unstable: alpha*dt/dx^2 = " + format("%.4g", edited.fourier()) + ", above " +
                   format("%.4g", edited.maxStableFourier()) + ". Use a time step of " + format("%.4g", maxStep) + " s or less.";
        }
        return {};
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

    std::shared_ptr<steakui::Disclosure> makeSection(const char* title){
        auto section = std::make_shared<steakui::Disclosure>(title, headingFont, GetTheme().font_size);
        section->SetOnToggled([this](){
            Layout(GetLayoutContext());
            PostRedraw();
        });
        return section;
    }

    std::shared_ptr<gui::Widget> buildPanel(){
        const int em = GetTheme().font_size;
        panel = std::make_shared<gui::Vert>(em / 2, gui::Margins(em));
        fields.reserve(24);

        panel->AddChild(std::make_shared<gui::Label>("Temperature (F)"));
        for (int row = 0; row < LEGEND_ROWS; ++row){
            const double t = 1.0 - static_cast<double>(row) / (LEGEND_ROWS - 1);
            auto line = std::make_shared<gui::Horiz>(em / 2);
            line->AddChild(std::make_shared<steakui::Chrome>(makeSwatch(heatColor(t), 2 * em, em)));
            legendLabels[row] = std::make_shared<gui::Label>("");
            line->AddChild(legendLabels[row]);
            panel->AddChild(line);
        }
        updateLegend();

        auto rangeRow = std::make_shared<gui::Horiz>(em);
        rangeRow->AddChild(std::make_shared<gui::Label>("Color max"));
        rangeSlider = std::make_shared<steakui::TrackSlider>();
        rangeSlider->SetLimits(coldTemp + 10.0, std::max(COLOR_MAX_F, edited.panTemp));
        rangeSlider->SetValue(hotTemp);
        rangeSlider->SetOnValueChanged([this](double value){ setColorMax(static_cast<float>(value)); });
        rangeRow->AddChild(rangeSlider);
        panel->AddChild(rangeRow);
        panel->AddFixed(em / 2);

        auto resetButton = std::make_shared<gui::Button>("Reset simulation");
        resetButton->SetOnClicked([this](){ askReset(); });
        panel->AddChild(resetButton);
        panel->AddFixed(em / 2);

        auto settings = std::make_shared<gui::Label>("Settings");
        settings->SetFontId(headingFont);
        settings->SetTextColor(gui::Color(1.0f, 1.0f, 1.0f));
        panel->AddChild(settings);
        auto hint = std::make_shared<gui::Label>("Type a value, then reset to run the simulation.");
        hint->SetTextColor(gui::Color(0.7f, 0.7f, 0.7f));
        panel->AddChild(hint);

        auto scroll = std::make_shared<gui::ScrollableVert>(em / 2, gui::Margins(0, em / 3, em + em / 2, 0));

        auto view = makeSection("View");
        auto& viewBody = view->content();
        panCheckbox = std::make_shared<gui::Checkbox>("Show pan");
        panCheckbox->SetChecked(showPan);
        panCheckbox->SetOnChecked([this](bool checked){ setShowPan(checked); });
        viewBody.AddChild(panCheckbox);
        singleSlice = std::make_shared<gui::Checkbox>("Single slice");
        singleSlice->SetOnChecked([this](bool checked){ setSlice(checked ? static_cast<int>(grid.dims[sliceAxis]) / 2 : -1); });
        viewBody.AddChild(singleSlice);
        sliceSlider = std::make_shared<steakui::TrackSlider>();
        sliceSlider->SetLimits(0, grid.dims[sliceAxis] - 1);
        sliceSlider->SetEnabled(false);
        sliceSlider->SetOnValueChanged([this](double value){ setSlice(static_cast<int>(value)); });
        viewBody.AddChild(sliceSlider);
        axisLabel = std::make_shared<gui::Label>("Slice Axis");
        axisLabel->SetTextColor(gui::Color(0.45f, 0.45f, 0.45f));
        axisCombo = std::make_shared<gui::Combobox>(std::vector<const char*>(AXIS_NAMES.begin(), AXIS_NAMES.end()));
        axisCombo->SetSelectedIndex(sliceAxis);
        axisCombo->SetEnabled(false);
        axisCombo->SetOnValueChanged([this](const char*, int index){ setSliceAxis(index); });
        viewBody.AddChild(std::make_shared<steakui::SliceAxisRow>(axisLabel, axisCombo));
        scroll->AddChild(view);

        auto time = makeSection("Time");
        auto& timeBody = time->content();
        addField(timeBody, "Duration (s)", true, [this](){ return format("%.6g", edited.duration); }, [this](const std::string& text, std::string& error){
            double value = 0.0;
            return parseNumber(text.c_str(), value, error) && commitDurationOrStep(value, edited.timeStep, error);
        });
        addField(timeBody, "Time step (s)", true, [this](){ return format("%.6g", edited.timeStep); }, [this](const std::string& text, std::string& error){
            double value = 0.0;
            return parseNumber(text.c_str(), value, error) && commitDurationOrStep(edited.duration, value, error);
        });
        addField(timeBody, "Steps", true, [this](){ return std::to_string(editedSteps); }, [this](const std::string& text, std::string& error){
            long count = 0;
            return parseCount(text.c_str(), count, 1, MAX_STEPS, error, "Steps") && commitSteps(count, error);
        });
        scroll->AddChild(time);

        auto gridSection = makeSection("Grid");
        auto& gridBody = gridSection->content();
        const char* dimNames[] = {"Cells X", "Cells Y", "Cells Z"};
        for (int axis = 0; axis < 3; ++axis){
            addField(gridBody, dimNames[axis], !gridFollowsMesh,
                     [this, axis](){
                         const glm::uvec3 dims = gridFollowsMesh ? grid.dims : edited.dims;
                         return std::to_string(dims[axis]);
                     },
                     [this, axis](const std::string& text, std::string& error){
                         long count = 0;
                         if (!parseCount(text.c_str(), count, 1, MAX_DIM, error, "Cells")) return false;
                         edited.dims[axis] = static_cast<uint32_t>(count);
                         return true;
                     });
        }
        addField(gridBody, "Cell size (mm)", true, [this](){ return format("%.6g", edited.spacing * MM_PER_M); },
                 [this](const std::string& text, std::string& error){
                     double mm = 0.0;
                     if (!parseNumber(text.c_str(), mm, error)) return false;
                     if (!(mm > 0.0)){
                         error = "Cell size must be positive.";
                         return false;
                     }
                     edited.spacing = static_cast<float>(mm / MM_PER_M);
                     return true;
                 });
        addField(gridBody, "Solid cells", false, [this](){ return std::to_string(solidCells); }, {});
        addField(gridBody, "On the pan", false, [this](){ return std::to_string(panContactCells); }, {});
        scroll->AddChild(gridSection);

        auto pan = makeSection("Pan");
        auto& panBody = pan->content();
        addField(panBody, "Temperature (F)", true, [this](){ return format("%.6g", edited.panTemp); }, [this](const std::string& text, std::string& error){
            double value = 0.0;
            if (!parseNumber(text.c_str(), value, error)) return false;
            edited.panTemp = static_cast<float>(value);
            return true;
        });
        addField(panBody, "Heat transfer (W/m^2K)", true, [this](){ return format("%.6g", edited.panTransfer); },
                 [this](const std::string& text, std::string& error){
                     double value = 0.0;
                     if (!parseNumber(text.c_str(), value, error)) return false;
                     if (value < 0.0){
                         error = "Heat transfer cannot be negative.";
                         return false;
                     }
                     edited.panTransfer = static_cast<float>(value);
                     return true;
                 });
        scroll->AddChild(pan);

        auto air = makeSection("Air");
        auto& airBody = air->content();
        addField(airBody, "Temperature (F)", true, [this](){ return format("%.6g", edited.airTemp); }, [this](const std::string& text, std::string& error){
            double value = 0.0;
            if (!parseNumber(text.c_str(), value, error)) return false;
            edited.airTemp = static_cast<float>(value);
            return true;
        });
        addField(airBody, "Heat transfer (W/m^2K)", true, [this](){ return format("%.6g", edited.airTransfer); },
                 [this](const std::string& text, std::string& error){
                     double value = 0.0;
                     if (!parseNumber(text.c_str(), value, error)) return false;
                     if (value < 0.0){
                         error = "Heat transfer cannot be negative.";
                         return false;
                     }
                     edited.airTransfer = static_cast<float>(value);
                     return true;
                 });
        scroll->AddChild(air);

        auto meat = makeSection("Meat");
        auto& meatBody = meat->content();
        addField(meatBody, "Start temperature (F)", true, [this](){ return format("%.6g", edited.initialTemp); },
                 [this](const std::string& text, std::string& error){
                     double value = 0.0;
                     if (!parseNumber(text.c_str(), value, error)) return false;
                     edited.initialTemp = static_cast<float>(value);
                     return true;
                 });
        addField(meatBody, "Temperature range (F)", false, [this](){
            if (solidCells == 0) return std::string("n/a");
            return format("%.4g", minTemp) + " to " + format("%.4g", maxTemp);
        }, {});
        addField(meatBody, "Diffusivity (m^2/s)", true, [this](){ return format("%.6g", edited.diffusivity); },
                 [this](const std::string& text, std::string& error){
                     double value = 0.0;
                     if (!parseNumber(text.c_str(), value, error)) return false;
                     if (!(value > 0.0)){
                         error = "Diffusivity must be positive.";
                         return false;
                     }
                     edited.diffusivity = static_cast<float>(value);
                     return true;
                 });
        addField(meatBody, "Conductivity (W/mK)", true, [this](){ return format("%.6g", edited.conductivity); },
                 [this](const std::string& text, std::string& error){
                     double value = 0.0;
                     if (!parseNumber(text.c_str(), value, error)) return false;
                     if (!(value > 0.0)){
                         error = "Conductivity must be positive.";
                         return false;
                     }
                     edited.conductivity = static_cast<float>(value);
                     return true;
                 });
        addField(meatBody, "Fourier r", false, [this](){ return format("%.4g", edited.fourier()); }, {});
        scroll->AddChild(meat);

        auto help = std::make_shared<gui::Label>("Drag to rotate, scroll to zoom\n Press 'X', 'Y', 'Z' to take slices along axes and move them with the arrow keys\n Press 'A' to toggle back to the full domain. \n Press 'P' to toggle pan.");
        help->SetTextColor(gui::Color(0.7f, 0.7f, 0.7f));
        scroll->AddChild(help);
        panel->AddChild(scroll);
        return panel;
    }

    void askReset(){
        if (resetting) return;
        const std::string error = resetError();
        if (!error.empty()){
            ShowMessageBox("Cannot reset", error.c_str());
            return;
        }
        const int width = std::max(320, 28 * GetTheme().font_size);
        auto dialog = std::make_shared<steakui::ConfirmDialog>(
            "Reset simulation", "Reset the simulation with the saved settings?", width, [this](bool accepted){
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
        progressDialog = std::make_shared<steakui::ResetProgressDialog>("Resetting the simulation.", width);
        progressDialog->SetFraction(0.0f);
        steakui::chromeHidden() = true;
        ShowDialog(progressDialog);

        const HeatParams ran = edited;
        const uint32_t ranSteps = editedSteps;
        const int axis = sliceAxis;
        const int sliceNow = slice;
        const bool pan = showPan;
        const float cold = coldTemp;
        const float hot = hotTemp;
        auto dialog = progressDialog;
        auto simulator = simulate;
        gui::Application::GetInstance().RunInThread([this, dialog, ran, ranSteps, axis, sliceNow, pan, cold, hot, simulator](){
            int lastPercent = -1;
            auto report = [this, dialog, &lastPercent](float fraction){
                const int percent = std::clamp(static_cast<int>(std::lround(fraction * 100.0f)), 0, 100);
                if (percent == lastPercent) return;
                lastPercent = percent;
                gui::Application::GetInstance().PostToMainThread(this, [dialog, percent](){ dialog->SetFraction(percent / 100.0f); });
            };
            try{
                report(0.01f);
                HeatGrid next = simulator(ran, ranSteps, [&](float fraction){ report(0.02f + 0.78f * fraction); });
                auto mesh = std::make_shared<open3d::geometry::TriangleMesh>(
                    buildHeatMesh(next, axis, sliceNow, pan, cold, hot, [&](float fraction){ report(0.80f + 0.19f * fraction); }));
                auto gridPtr = std::make_shared<HeatGrid>(std::move(next));
                auto applied = std::make_shared<HeatParams>(ran);
                gui::Application::GetInstance().PostToMainThread(
                    this, [this, gridPtr, applied, ranSteps, axis, sliceNow, mesh](){ finishReset(*gridPtr, *applied, ranSteps, axis, sliceNow, *mesh); });
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

    void finishReset(HeatGrid next, HeatParams ran, uint32_t ranSteps, int axis, int sliceNow, open3d::geometry::TriangleMesh mesh){
        const bool shapeChanged = next.dims != grid.dims || next.spacing != grid.spacing;
        grid = std::move(next);
        edited = ran;
        edited.dims = grid.dims;
        editedSteps = ranSteps;
        recount();

        sliceAxis = axis;
        slice = sliceNow < 0 ? -1 : std::min(sliceNow, std::max(0, static_cast<int>(grid.dims[sliceAxis]) - 1));
        blocks = std::move(mesh);
        uploadBlocks();
        if (shapeChanged && !blocks.triangles_.empty()){
            const auto& bounds = scene->GetScene()->GetBoundingBox();
            scene->SetupCamera(60.0f, bounds, bounds.GetCenter().cast<float>());
        }

        axisCombo->SetSelectedIndex(sliceAxis);
        sliceSlider->SetLimits(0, std::max(1u, grid.dims[sliceAxis]) - 1u);
        syncSliceControls();
        const double upper = std::max({static_cast<double>(COLOR_MAX_F), static_cast<double>(edited.panTemp), static_cast<double>(hotTemp),
                                       coldTemp + 10.0});
        rangeSlider->SetLimits(coldTemp + 10.0, upper);
        refreshFields();
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
        const int middle = static_cast<int>(grid.dims[sliceAxis]) / 2;
        switch (e.key){
            case gui::KEY_X: setSliceAxis(0); return true;
            case gui::KEY_Y: setSliceAxis(1); return true;
            case gui::KEY_Z: setSliceAxis(2); return true;
            case gui::KEY_UP: setSlice(slice < 0 ? middle : slice + 1); return true;
            case gui::KEY_DOWN: setSlice(slice < 0 ? middle : std::max(slice - 1, 0)); return true;
            case gui::KEY_A: setSlice(-1); return true;
            case gui::KEY_P: setShowPan(!showPan); return true;
            default: return false;
        }
    }

    HeatGrid grid;
    HeatParams edited;
    HeatSimulator simulate;
    uint32_t editedSteps = 0;
    bool gridFollowsMesh = false;
    gui::FontId headingFont = 0;
    float coldTemp = 0.0f, hotTemp = 0.0f;
    float minTemp = 0.0f, maxTemp = 0.0f;
    size_t solidCells = 0, panContactCells = 0;
    int sliceAxis = 2;
    int slice = -1;
    bool showPan = true;
    bool resetting = false;
    bool progressArmed = false;
    double progressDoneAt = 0.0;
    open3d::geometry::TriangleMesh blocks;
    rendering::MaterialRecord material;
    std::shared_ptr<gui::SceneWidget> scene;
    std::shared_ptr<gui::Vert> panel;
    std::array<std::shared_ptr<gui::Label>, LEGEND_ROWS> legendLabels;
    std::shared_ptr<gui::Checkbox> panCheckbox;
    std::shared_ptr<gui::Label> axisLabel;
    std::shared_ptr<gui::Combobox> axisCombo;
    std::shared_ptr<gui::Checkbox> singleSlice;
    std::shared_ptr<steakui::TrackSlider> sliceSlider;
    std::shared_ptr<steakui::TrackSlider> rangeSlider;
    std::shared_ptr<steakui::ResetProgressDialog> progressDialog;
    std::vector<SettingField> fields;
};

gui::FontId headingFont(){
    try{
        return gui::Application::GetInstance().AddFont(gui::FontDescription(gui::FontDescription::SANS_SERIF, gui::FontStyle::BOLD, 20));
    } catch (const std::exception& e){
        std::cerr << "heat view: heading font unavailable (" << e.what() << ")\n";
        return gui::Application::DEFAULT_FONT_ID;
    }
}

}  // namespace

void printCenterColumn(const HeatGrid& grid, uint32_t rows){
    const uint32_t cx = grid.dims.x / 2, cz = grid.dims.z / 2;
    std::cout << "Temperature up the center column at t = " << grid.time << " s:\n";
    for (uint32_t y = 0; y < std::min(grid.dims.y, rows); ++y){
        const uint32_t kind = grid.kind(cx, y, cz);
        std::cout << "  y = " << format("%5.1f", y * grid.spacing * MM_PER_M) << " mm: " << format("%6.1f", grid.at(cx, y, cz)) << " F"
                  << (kind == CELL_PAN ? "  (pan)" : kind == CELL_AIR ? "  (air)" : "") << "\n";
    }
    std::cout << std::flush;
}

void showHeatGrid(const std::string& title, HeatGrid grid, const HeatParams& params, uint32_t steps, HeatSimulator simulate,
                  bool gridFollowsMesh){
    auto& app = gui::Application::GetInstance();
    app.Initialize(OPEN3D_RESOURCE_DIR);
    const gui::FontId font = headingFont();
    app.AddWindow(std::make_shared<HeatViewWindow>(title, std::move(grid), params, steps, std::move(simulate), gridFollowsMesh, font));
    app.Run();
}
