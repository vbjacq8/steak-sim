#pragma once

// Confirm and progress dialogs shared by the heat and voxel viewers.

#include <open3d/geometry/Image.h>
#include <open3d/visualization/gui/Button.h>
#include <open3d/visualization/gui/Combobox.h>
#include <open3d/visualization/gui/Dialog.h>
#include <open3d/visualization/gui/Events.h>
#include <open3d/visualization/gui/ImageWidget.h>
#include <open3d/visualization/gui/Label.h>
#include <open3d/visualization/gui/Layout.h>
#include <open3d/visualization/gui/ProgressBar.h>
#include <open3d/visualization/gui/Theme.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <string>

namespace steakui {

namespace gui = open3d::visualization::gui;

// Image widgets are drawn by the 3D renderer, so they ignore ImGui clipping and show through dialogs.
inline bool& chromeHidden(){
    static bool hidden = false;
    return hidden;
}

class Chrome : public gui::Widget{
public:
    explicit Chrome(std::shared_ptr<gui::Widget> child){ AddChild(std::move(child)); }

    gui::Size CalcPreferredSize(const gui::LayoutContext& context, const Constraints& constraints) const override{
        const auto children = GetChildren();
        return children.empty() ? gui::Size() : children.front()->CalcPreferredSize(context, constraints);
    }

    void Layout(const gui::LayoutContext& context) override{
        const auto children = GetChildren();
        if (!children.empty()) children.front()->SetFrame(GetFrame());
        gui::Widget::Layout(context);
    }

    DrawResult Draw(const gui::DrawContext& context) override{
        if (chromeHidden()) return DrawResult::NONE;
        return gui::Widget::Draw(context);
    }
};

// Bold title with a show/hide button. The body is removed from layout when hidden, so nothing is left behind.
class Disclosure : public gui::Vert{
public:
    Disclosure(const char* title, gui::FontId font, int em) : gui::Vert(std::max(1, em / 3)){
        auto header = std::make_shared<gui::Horiz>(std::max(1, em / 2));
        auto label = std::make_shared<gui::Label>(title);
        label->SetFontId(font);
        label->SetTextColor(gui::Color(1.0f, 1.0f, 1.0f));
        toggle_ = std::make_shared<gui::Button>("Hide");
        header->AddChild(label);
        header->AddStretch();
        header->AddChild(toggle_);
        content_ = std::make_shared<gui::Vert>(std::max(1, em / 3), gui::Margins(em, em / 5, 0, em / 3));
        AddChild(header);
        AddChild(content_);
        toggle_->SetOnClicked([this](){
            const bool show = !content_->IsVisible();
            content_->SetVisible(show);
            toggle_->SetText(show ? "Hide" : "Show");
            if (onToggled_) onToggled_();
        });
    }

    gui::Widget& content(){ return *content_; }
    void SetOnToggled(std::function<void()> onToggled){ onToggled_ = std::move(onToggled); }

private:

    std::shared_ptr<gui::Button> toggle_;
    std::shared_ptr<gui::Vert> content_;
    std::function<void()> onToggled_;
};

// Label text sits at the top of its frame; the combobox text is inset by ImGui's frame padding.
// This row drops the label by that padding so "Slice Axis" lines up with "X" / "Y" / "Z".
class SliceAxisRow : public gui::Widget{
public:
    SliceAxisRow(std::shared_ptr<gui::Label> label, std::shared_ptr<gui::Combobox> combo)
        : label_(std::move(label)), combo_(std::move(combo)){
        AddChild(label_);
        AddChild(combo_);
    }

    gui::Size CalcPreferredSize(const gui::LayoutContext& context, const Constraints& constraints) const override{
        const auto label = label_->CalcPreferredSize(context, constraints);
        const auto combo = combo_->CalcPreferredSize(context, constraints);
        const int width = constraints.width >= DIM_GROW ? DIM_GROW : constraints.width;
        return gui::Size(width, std::max(label.height, combo.height));
    }

    void Layout(const gui::LayoutContext& context) override{
        const gui::Rect frame = GetFrame();
        const auto labelSize = label_->CalcPreferredSize(context, gui::Widget::Constraints());
        const auto comboSize = combo_->CalcPreferredSize(context, gui::Widget::Constraints());
        const int gap = std::max(4, context.theme.font_size / 2);
        const int comboWidth = std::min(comboSize.width, std::max(0, frame.width - labelSize.width - gap));
        const int textHeight = context.theme.font_size;
        const int labelY = frame.y + std::max(0, (comboSize.height - textHeight) / 2);
        label_->SetFrame(gui::Rect(frame.x, labelY, labelSize.width, textHeight));
        combo_->SetFrame(gui::Rect(frame.x + labelSize.width + gap, frame.y, comboWidth, comboSize.height));
        gui::Widget::Layout(context);
    }

private:
    std::shared_ptr<gui::Label> label_;
    std::shared_ptr<gui::Combobox> combo_;
};

inline std::shared_ptr<gui::ImageWidget> solidSwatch(uint8_t red, uint8_t green, uint8_t blue){
    auto image = std::make_shared<open3d::geometry::Image>();
    image->Prepare(1, 1, 3, 1);
    image->data_[0] = red;
    image->data_[1] = green;
    image->data_[2] = blue;
    auto swatch = std::make_shared<gui::ImageWidget>(image);
    swatch->GetUIImage()->SetScaling(gui::UIImage::Scaling::ANY);
    return swatch;
}

// A slider with a dark trough, a light outline, and a fill up to the knob. The value is drawn on the
// track and again in a tooltip while the pointer is over it.
class TrackSlider : public gui::Widget{
public:
    TrackSlider(){
        outline_ = solidSwatch(196, 202, 214);
        wellSwatch_ = solidSwatch(22, 24, 28);
        fill_ = solidSwatch(70, 132, 186);
        knobSwatch_ = solidSwatch(244, 244, 246);
        outlineOff_ = solidSwatch(110, 112, 118);
        wellOff_ = solidSwatch(36, 38, 42);
        fillOff_ = solidSwatch(68, 70, 76);
        knobOff_ = solidSwatch(130, 132, 138);
        valueLabel_ = std::make_shared<gui::Label>("0");
        valueLabel_->SetTextColor(gui::Color(1.0f, 1.0f, 1.0f));
        AddChild(outline_);
        AddChild(wellSwatch_);
        AddChild(fill_);
        AddChild(knobSwatch_);
        AddChild(outlineOff_);
        AddChild(wellOff_);
        AddChild(fillOff_);
        AddChild(knobOff_);
        AddChild(valueLabel_);
        showEnabled(true);
    }

    void SetLimits(double minValue, double maxValue){
        min_ = std::min(minValue, maxValue);
        max_ = std::max(minValue, maxValue);
        SetValue(value_);
    }

    void SetValue(double value){
        value_ = std::round(std::clamp(value, min_, max_));
        place();
        updateText();
    }

    void SetEnabled(bool enabled) override{
        gui::Widget::SetEnabled(enabled);
        showEnabled(enabled);
    }

    void SetOnValueChanged(std::function<void(double)> onChanged){ onChanged_ = std::move(onChanged); }

    gui::Size CalcPreferredSize(const gui::LayoutContext& context, const Constraints& constraints) const override{
        const int width = constraints.width >= DIM_GROW ? DIM_GROW : constraints.width;
        return gui::Size(width, std::max(8, context.theme.font_size + context.theme.font_size / 3));
    }

    void Layout(const gui::LayoutContext& context) override{
        gui::Widget::Layout(context);
        const gui::Rect frame = GetFrame();
        const int border = std::max(2, context.theme.font_size / 8);
        outline_->SetFrame(frame);
        outlineOff_->SetFrame(frame);
        well_ = gui::Rect(frame.x + border, frame.y + border, std::max(0, frame.width - 2 * border), std::max(0, frame.height - 2 * border));
        wellSwatch_->SetFrame(well_);
        wellOff_->SetFrame(well_);
        knob_ = std::clamp(std::max(border * 3, context.theme.font_size / 2), 1, std::max(1, well_.width));
        em_ = context.theme.font_size;
        place();
        updateText();
    }

    gui::Widget::EventResult Mouse(const gui::MouseEvent& e) override{
        const bool inside = GetFrame().Contains(e.x, e.y);
        const bool drag = e.type == gui::MouseEvent::DRAG || e.type == gui::MouseEvent::BUTTON_UP;
        if (!inside && !drag) return gui::Widget::EventResult::IGNORED;
        if (!IsEnabled()) return gui::Widget::EventResult::DISCARD;
        if (e.type == gui::MouseEvent::BUTTON_DOWN && e.button.button != gui::MouseButton::LEFT) return gui::Widget::EventResult::IGNORED;
        if (e.type != gui::MouseEvent::BUTTON_DOWN && e.type != gui::MouseEvent::DRAG && e.type != gui::MouseEvent::BUTTON_UP)
            return gui::Widget::EventResult::IGNORED;
        if (e.type != gui::MouseEvent::BUTTON_UP) setFromX(e.x);
        return gui::Widget::EventResult::CONSUMED;
    }

    gui::Widget::DrawResult Draw(const gui::DrawContext& context) override{
        if (chromeHidden()) return gui::Widget::DrawResult::NONE;
        return gui::Widget::Draw(context);
    }

private:
    void showEnabled(bool enabled){
        outline_->SetVisible(enabled);
        wellSwatch_->SetVisible(enabled);
        fill_->SetVisible(enabled);
        knobSwatch_->SetVisible(enabled);
        outlineOff_->SetVisible(!enabled);
        wellOff_->SetVisible(!enabled);
        fillOff_->SetVisible(!enabled);
        knobOff_->SetVisible(!enabled);
        valueLabel_->SetVisible(enabled);
    }

    void updateText(){
        const std::string text = std::to_string(static_cast<long long>(std::llround(value_)));
        valueLabel_->SetText(text.c_str());
        valueLabel_->SetTooltip(text.c_str());
        const gui::Rect frame = GetFrame();
        if (frame.width <= 0 || em_ <= 0) return;
        const int box = std::max(em_, static_cast<int>(text.size()) * em_);
        const int height = em_ + em_ / 3;
        valueLabel_->SetFrame(gui::Rect(frame.x + (frame.width - box) / 2, frame.y + (frame.height - height) / 2, box, height));
    }

    void place(){
        if (well_.width <= 0) return;
        const int span = std::max(1, well_.width - knob_);
        const double t = max_ > min_ ? std::clamp((value_ - min_) / (max_ - min_), 0.0, 1.0) : 0.0;
        const int x = well_.x + static_cast<int>(std::round(t * span));
        const gui::Rect fill(well_.x, well_.y, std::clamp(x - well_.x + knob_ / 2, 0, well_.width), well_.height);
        const gui::Rect knob(x, well_.y, knob_, well_.height);
        fill_->SetFrame(fill);
        fillOff_->SetFrame(fill);
        knobSwatch_->SetFrame(knob);
        knobOff_->SetFrame(knob);
    }

    void setFromX(int x){
        const int span = std::max(1, well_.width - knob_);
        const double t = std::clamp((x - well_.x - knob_ / 2) / static_cast<double>(span), 0.0, 1.0);
        const double value = std::round(min_ + t * (max_ - min_));
        if (value == value_) return;
        value_ = value;
        place();
        updateText();
        if (onChanged_) onChanged_(value_);
    }

    double min_ = 0.0;
    double max_ = 1.0;
    double value_ = 0.0;
    int knob_ = 1;
    int em_ = 0;
    gui::Rect well_;
    std::function<void(double)> onChanged_;
    std::shared_ptr<gui::ImageWidget> outline_;
    std::shared_ptr<gui::ImageWidget> wellSwatch_;
    std::shared_ptr<gui::ImageWidget> fill_;
    std::shared_ptr<gui::ImageWidget> knobSwatch_;
    std::shared_ptr<gui::ImageWidget> outlineOff_;
    std::shared_ptr<gui::ImageWidget> wellOff_;
    std::shared_ptr<gui::ImageWidget> fillOff_;
    std::shared_ptr<gui::ImageWidget> knobOff_;
    std::shared_ptr<gui::Label> valueLabel_;
};

// Open3D's progress bar is a few pixels tall. This gives it a text-line of height.
class ThickProgress : public gui::Widget{
public:
    ThickProgress(){ AddChild(bar_); }

    void SetFraction(float value){ bar_->SetValue(std::clamp(value, 0.0f, 1.0f)); }

    gui::Size CalcPreferredSize(const gui::LayoutContext& context, const Constraints& constraints) const override{
        const int width = constraints.width >= DIM_GROW ? 280 : constraints.width;
        return gui::Size(width, std::max(8, context.theme.font_size));
    }

    void Layout(const gui::LayoutContext& context) override{
        bar_->SetFrame(GetFrame());
        gui::Widget::Layout(context);
    }

private:
    std::shared_ptr<gui::ProgressBar> bar_ = std::make_shared<gui::ProgressBar>();
};

class ConfirmDialog : public gui::Dialog{
public:
    ConfirmDialog(const char* title, const char* message, int width, std::function<void(bool)> onDone)
        : gui::Dialog(title), onDone_(std::move(onDone)){
        auto body = std::make_shared<gui::Vert>(8, gui::Margins(width / 28));
        body->SetPreferredWidth(width);
        body->AddChild(std::make_shared<gui::Label>(message));
        body->AddFixed(8);
        auto row = std::make_shared<gui::Horiz>(8);
        row->AddStretch();
        auto cancel = std::make_shared<gui::Button>("Cancel");
        auto accept = std::make_shared<gui::Button>("Reset");
        cancel->SetOnClicked([this](){ onDone_(false); });
        accept->SetOnClicked([this](){ onDone_(true); });
        row->AddChild(cancel);
        row->AddChild(accept);
        body->AddChild(row);
        AddChild(body);
    }

private:
    std::function<void(bool)> onDone_;
};

class ResetProgressDialog : public gui::Dialog{
public:
    explicit ResetProgressDialog(const char* message, int width) : gui::Dialog("Resetting"){
        auto body = std::make_shared<gui::Vert>(8, gui::Margins(width / 28));
        body->SetPreferredWidth(width);
        body->AddChild(std::make_shared<gui::Label>(message));
        bar_ = std::make_shared<ThickProgress>();
        body->AddChild(bar_);
        percent_ = std::make_shared<gui::Label>("0%");
        body->AddChild(percent_);
        AddChild(body);
    }

    void SetFraction(float value){
        value = std::clamp(value, 0.0f, 1.0f);
        bar_->SetFraction(value);
        percent_->SetText((std::to_string(static_cast<int>(std::lround(value * 100.0f))) + "%").c_str());
    }

private:
    std::shared_ptr<ThickProgress> bar_;
    std::shared_ptr<gui::Label> percent_;
};

}  // namespace steakui
