// Heat-map window shared by heat_viewer and steak_heat_demo. Only the faces between a drawn cell and
// an undrawn one (air, hidden pan, outside the slice, or outside the grid) are built, so a 128^3 grid
// stays at ~100k quads.

#include "heat_view.hpp"

#include <open3d/Open3D.h>
#include <open3d/visualization/rendering/ColorGrading.h>
#include <open3d/visualization/rendering/View.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>

namespace gui = open3d::visualization::gui;
namespace rendering = open3d::visualization::rendering;

namespace {

// Blocks are drawn slightly smaller than the cell so the background shows the grid between them.
constexpr double BLOCK_SHRINK = 0.9;
constexpr double MM_PER_M = 1000.0;
constexpr int LEGEND_ROWS = 7;
// Default color range: room temperature to just past well done (medium well is ~160 F).
constexpr float COLOR_MIN_F = 70.0f;
constexpr float COLOR_MAX_F = 200.0f;
constexpr std::array<const char*, 3> AXIS_NAMES{"X", "Y", "Z"};

std::string format(const char* fmt, double value){
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), fmt, value);
    return buffer;
}

// Blue (cold) -> cyan -> yellow -> orange -> red (hot); t in [0, 1].
Eigen::Vector3d heatColor(double t){
    static constexpr std::array<std::array<double, 3>, 5> stops{{
        {0.12, 0.25, 0.85}, {0.10, 0.75, 0.90}, {0.98, 0.88, 0.15}, {0.98, 0.45, 0.05}, {0.75, 0.05, 0.05}}};
    const double x = std::clamp(t, 0.0, 1.0) * (stops.size() - 1);
    const size_t i = std::min(static_cast<size_t>(x), stops.size() - 2);
    const double f = x - i;
    return {stops[i][0] + f * (stops[i + 1][0] - stops[i][0]), stops[i][1] + f * (stops[i + 1][1] - stops[i][1]),
            stops[i][2] + f * (stops[i + 1][2] - stops[i][2])};
}

// Labels ignore SetBackgroundColor, so legend swatches are solid-color images.
std::shared_ptr<gui::ImageWidget> makeSwatch(const Eigen::Vector3d& color, int width, int height){
    auto image = std::make_shared<open3d::geometry::Image>();
    image->Prepare(width, height, 3, 1);
    for (int i = 0; i < width * height; ++i){
        for (int c = 0; c < 3; ++c) image->data_[3 * i + c] = static_cast<uint8_t>(std::lround(255.0 * color[c]));
    }
    return std::make_shared<gui::ImageWidget>(image);
}

class HeatViewWindow : public gui::Window{
public:
    HeatViewWindow(const std::string& title, HeatGrid gridIn, HeatParams paramsIn, uint32_t stepsIn)
        : gui::Window(title, 1400, 850), grid(std::move(gridIn)), params(paramsIn), steps(stepsIn){
        coldTemp = COLOR_MIN_F;
        hotTemp = COLOR_MAX_F;
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

        scene = std::make_shared<gui::SceneWidget>();
        scene->SetScene(std::make_shared<rendering::Open3DScene>(GetRenderer()));
        scene->GetScene()->SetBackground({0.08f, 0.08f, 0.10f, 1.0f});
        // The default tone mapping washes out unlit colors; with linear tone mapping and linearized vertex
        // colors, blocks show the same colors as the legend.
        scene->GetScene()->GetView()->SetColorGrading(rendering::ColorGradingParams(
            rendering::ColorGradingParams::Quality::kHigh, rendering::ColorGradingParams::ToneMapping::kLinear));
        material.shader = "defaultUnlit";
        rebuildBlocks();

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
    bool inRegion(glm::ivec3 cell) const{
        return glm::all(glm::greaterThanEqual(cell, glm::ivec3(regionLo))) && glm::all(glm::lessThanEqual(cell, glm::ivec3(regionHi)));
    }

    bool drawn(glm::ivec3 cell) const{
        if (!inRegion(cell)) return false;
        const uint32_t kind = grid.kind(cell.x, cell.y, cell.z);
        return kind == CELL_SOLID || (kind == CELL_PAN && showPan);
    }

    // One quad on the side of `cell` facing -axis or +axis.
    void appendFace(glm::uvec3 cell, int axis, bool positive){
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
        const Eigen::Vector3d color = heatColor(t).array().pow(2.2);  // sRGB -> linear
        for (const auto& [cu, cv] : corners){
            Eigen::Vector3d p;
            p[axis] = plane;
            p[u] = cu;
            p[v] = cv;
            blocks.vertices_.push_back(p);
            blocks.vertex_colors_.push_back(color);
        }
        // Counter-clockwise seen from outside: e_u x e_v = +e_axis.
        if (positive){
            blocks.triangles_.emplace_back(base, base + 1, base + 2);
            blocks.triangles_.emplace_back(base, base + 2, base + 3);
        } else{
            blocks.triangles_.emplace_back(base, base + 2, base + 1);
            blocks.triangles_.emplace_back(base, base + 3, base + 2);
        }
    }

    // The whole grid, or one slice when slice >= 0.
    void rebuildBlocks(){
        regionLo = glm::uvec3(0);
        regionHi = grid.dims - 1u;
        if (slice >= 0) regionLo[sliceAxis] = regionHi[sliceAxis] = static_cast<uint32_t>(slice);

        blocks.Clear();
        for (uint32_t z = regionLo.z; z <= regionHi.z; ++z){
            for (uint32_t y = regionLo.y; y <= regionHi.y; ++y){
                for (uint32_t x = regionLo.x; x <= regionHi.x; ++x){
                    const glm::ivec3 cell(x, y, z);
                    if (!drawn(cell)) continue;
                    for (int axis = 0; axis < 3; ++axis){
                        for (int side : {-1, 1}){
                            glm::ivec3 neighbor = cell;
                            neighbor[axis] += side;
                            if (!drawn(neighbor)) appendFace(glm::uvec3(cell), axis, side > 0);
                        }
                    }
                }
            }
        }

        auto& scene3d = *scene->GetScene();
        if (scene3d.HasGeometry("blocks")) scene3d.RemoveGeometry("blocks");
        if (!blocks.triangles_.empty()) scene3d.AddGeometry("blocks", &blocks, material, false);
        PostRedraw();
    }

    void setSlice(int newSlice){
        slice = std::clamp(newSlice, -1, static_cast<int>(grid.dims[sliceAxis]) - 1);
        singleSlice->SetChecked(slice >= 0);
        sliceSlider->SetEnabled(slice >= 0);
        if (slice >= 0){
            sliceSlider->SetValue(slice);
            sliceLabel->SetText(format("%.1f mm", (slice + 0.5) * grid.spacing * MM_PER_M).c_str());
        } else{
            sliceLabel->SetText("");
        }
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

    // Temperatures at or above `value` get the hottest color.
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

    std::shared_ptr<gui::Widget> buildPanel(){
        const int em = GetTheme().font_size;
        panel = std::make_shared<gui::Vert>(em / 4, gui::Margins(em));

        auto title = std::make_shared<gui::Label>("Temperature (F)");
        title->SetTextColor(gui::Color(1.0f, 1.0f, 1.0f));
        panel->AddChild(title);
        for (int row = 0; row < LEGEND_ROWS; ++row){
            const double t = 1.0 - static_cast<double>(row) / (LEGEND_ROWS - 1);
            const Eigen::Vector3d color = heatColor(t);
            auto line = std::make_shared<gui::Horiz>(em / 2);
            line->AddChild(makeSwatch(color, 3 * em, em));
            legendLabels[row] = std::make_shared<gui::Label>("");
            line->AddChild(legendLabels[row]);
            panel->AddChild(line);
        }
        updateLegend();

        auto rangeRow = std::make_shared<gui::Horiz>(em / 2);
        rangeRow->AddChild(std::make_shared<gui::Label>("color max"));
        auto rangeSlider = std::make_shared<gui::Slider>(gui::Slider::INT);
        rangeSlider->SetLimits(coldTemp + 10.0, std::max(COLOR_MAX_F, params.panTemp));
        rangeSlider->SetValue(hotTemp);
        rangeSlider->SetOnValueChanged([this](double value){ setColorMax(static_cast<float>(value)); });
        rangeRow->AddChild(rangeSlider);
        panel->AddChild(rangeRow);
        panel->AddFixed(em);

        panCheckbox = std::make_shared<gui::Checkbox>("pan cells");
        panCheckbox->SetChecked(showPan);
        panCheckbox->SetOnChecked([this](bool checked){ setShowPan(checked); });
        panel->AddChild(panCheckbox);

        auto axisRow = std::make_shared<gui::Horiz>(em / 2);
        axisRow->AddChild(std::make_shared<gui::Label>("slice axis"));
        axisCombo = std::make_shared<gui::Combobox>(std::vector<const char*>(AXIS_NAMES.begin(), AXIS_NAMES.end()));
        axisCombo->SetSelectedIndex(sliceAxis);
        axisCombo->SetOnValueChanged([this](const char*, int index){ setSliceAxis(index); });
        axisRow->AddChild(axisCombo);
        panel->AddChild(axisRow);

        singleSlice = std::make_shared<gui::Checkbox>("single slice");
        singleSlice->SetOnChecked([this](bool checked){ setSlice(checked ? static_cast<int>(grid.dims[sliceAxis]) / 2 : -1); });
        panel->AddChild(singleSlice);
        sliceSlider = std::make_shared<gui::Slider>(gui::Slider::INT);
        sliceSlider->SetLimits(0, grid.dims[sliceAxis] - 1);
        sliceSlider->SetEnabled(false);
        sliceSlider->SetOnValueChanged([this](double value){ setSlice(static_cast<int>(value)); });
        panel->AddChild(sliceSlider);
        sliceLabel = std::make_shared<gui::Label>("");
        panel->AddChild(sliceLabel);
        panel->AddFixed(em);

        const std::string info =
            "Grid " + std::to_string(grid.dims.x) + " x " + std::to_string(grid.dims.y) + " x " + std::to_string(grid.dims.z) +
            ", dx " + format("%.3g mm", grid.spacing * MM_PER_M) + "\n" +
            std::to_string(solidCells) + " solid cells, " + std::to_string(panContactCells) + " on the pan\n" +
            "t = " + format("%.1f s", grid.time) + " (" + std::to_string(steps) + " steps of " + format("%.3g s", params.timeStep) + ")\n" +
            "alpha " + format("%.3g m^2/s", params.diffusivity) + ", r = alpha dt/dx^2 = " + format("%.3f", params.fourier()) + "\n" +
            "pan (y = 0) at " + format("%.0f F", params.panTemp) + ", h = " + format("%.3g W/m^2K", params.panTransfer) +
            "\nair at " + format("%.0f F", params.airTemp) + ", h = " + format("%.3g W/m^2K", params.airTransfer) + "\n" +
            "solid min " + format("%.1f F", minTemp) + ", max " + format("%.1f F", maxTemp);
        panel->AddChild(std::make_shared<gui::Label>(info.c_str()));
        panel->AddFixed(em);

        auto keys = std::make_shared<gui::Label>(
            "Keys\n"
            "X / Y / Z   slice axis\n"
            "Up / Down   step slice\n"
            "A           whole grid\n"
            "P           toggle pan cells\n"
            "Mouse: drag rotate, scroll zoom,\n"
            "  shift-drag or right-drag pan");
        keys->SetTextColor(gui::Color(0.7f, 0.7f, 0.7f));
        panel->AddChild(keys);
        panel->AddStretch();
        return panel;
    }

    bool onKey(const gui::KeyEvent& e){
        if (e.type != gui::KeyEvent::DOWN) return false;
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
    HeatParams params;
    uint32_t steps;
    float coldTemp = 0.0f, hotTemp = 0.0f;  // color map range
    float minTemp = 0.0f, maxTemp = 0.0f;   // actual range over solid cells
    size_t solidCells = 0, panContactCells = 0;
    int sliceAxis = 2;
    int slice = -1;
    bool showPan = true;
    glm::uvec3 regionLo{}, regionHi{};
    open3d::geometry::TriangleMesh blocks;
    rendering::MaterialRecord material;
    std::shared_ptr<gui::SceneWidget> scene;
    std::shared_ptr<gui::Vert> panel;
    std::array<std::shared_ptr<gui::Label>, LEGEND_ROWS> legendLabels;
    std::shared_ptr<gui::Checkbox> panCheckbox;
    std::shared_ptr<gui::Combobox> axisCombo;
    std::shared_ptr<gui::Checkbox> singleSlice;
    std::shared_ptr<gui::Slider> sliceSlider;
    std::shared_ptr<gui::Label> sliceLabel;
};

}  // namespace

void printCenterColumn(const HeatGrid& grid, uint32_t rows){
    const uint32_t cx = grid.dims.x / 2, cz = grid.dims.z / 2;
    std::cout << "Temperature up the center column at t = " << grid.time << " s:\n";
    for (uint32_t y = 0; y < std::min(grid.dims.y, rows); ++y){
        const uint32_t kind = grid.kind(cx, y, cz);
        std::cout << "  y = " << format("%5.1f", y * grid.spacing * MM_PER_M) << " mm: "
                  << format("%6.1f", grid.at(cx, y, cz)) << " F"
                  << (kind == CELL_PAN ? "  (pan)" : kind == CELL_AIR ? "  (air)" : "") << "\n";
    }
    std::cout << std::flush;
}

void showHeatGrid(const std::string& title, HeatGrid grid, const HeatParams& params, uint32_t steps){
    auto& app = gui::Application::GetInstance();
    app.Initialize(OPEN3D_RESOURCE_DIR);
    app.AddWindow(std::make_shared<HeatViewWindow>(title, std::move(grid), params, steps));
    app.Run();
}
