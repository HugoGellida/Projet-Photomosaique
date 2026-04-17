#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFML/Window/Mouse.hpp>

class Button {
public:
    // Constructor to create a button
    Button(float x, float y, float width, float height, const std::string& label, sf::Font& font) {
        // Set up the button background
        buttonShape.setSize(sf::Vector2f(width, height));
        buttonShape.setPosition(x, y);
        buttonShape.setFillColor(restColor);
        buttonShape.setOutlineColor(outLine);
        buttonShape.setOutlineThickness(2.f);

        // Set up the text
        buttonText.setFont(font);
        buttonText.setString(label);
        buttonText.setCharacterSize(20);
        buttonText.setFillColor(outLine);
        buttonText.setPosition(
            x + (width - buttonText.getLocalBounds().width) / 2, 
            y + (height - buttonText.getLocalBounds().height - buttonText.getCharacterSize()/2) / 2
        );
    }

    // Draw the button on the window
    void draw(sf::RenderWindow& window) {
        window.draw(buttonShape);
        window.draw(buttonText);
    }

    // Check if the button was clicked
    bool isClicked(sf::Event& event, sf::RenderWindow& window) {
        // Check if the mouse is inside the button's bounds
        if (event.type == sf::Event::MouseButtonPressed) {
            if (event.mouseButton.button == sf::Mouse::Left) {
                if (buttonShape.getGlobalBounds().contains(window.mapPixelToCoords(sf::Vector2i(event.mouseButton.x, event.mouseButton.y)))) {
                    return true;
                }
            }
        }
        return false;
    }

    // Optional: Change button color on hover (mouse over)
    void updateHover(sf::RenderWindow& window) {
        if (buttonShape.getGlobalBounds().contains(window.mapPixelToCoords(sf::Mouse::getPosition(window)))) {
            buttonShape.setFillColor(hoverColor);
        } else {
            buttonShape.setFillColor(restColor);
        }
    }

    sf::Color restColor = sf::Color(128, 128, 128);
    sf::Color hoverColor = sf::Color(165, 165, 165);
    sf::Color outLine = sf::Color(255, 255, 255);
    sf::Text buttonText;

private:
    sf::RectangleShape buttonShape;
};