document.addEventListener('DOMContentLoaded', function() {
    // Scroll up button functionality
    const scrollUpBtn = document.querySelector('.scroll-up-btn');
    if (scrollUpBtn) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 300) {
                scrollUpBtn.classList.add('active');
            } else {
                scrollUpBtn.classList.remove('active');
            }
        });

        scrollUpBtn.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }

    // Sticky navbar on scroll
    const navbarBottom = document.querySelector('.navbar-bottom');
    if (navbarBottom) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 20) {
                navbarBottom.classList.add('sticky');
            } else {
                navbarBottom.classList.remove('sticky');
            }
        });
    }

    // Form validation for prediction form
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            const requiredInputs = predictionForm.querySelectorAll('select[required], input[required]');
            let valid = true;

            requiredInputs.forEach(input => {
                if (!input.value) {
                    input.classList.add('is-invalid');
                    valid = false;
                } else {
                    input.classList.remove('is-invalid');
                }
            });

            if (!valid) {
                e.preventDefault();
                alert('Please fill in all required fields.');
            }
        });
    }

    // CIBIL score live indicator
    const cibilInput = document.getElementById('cibil_score');
    const scoreLow = document.querySelector('.score-low');
    const scoreMedium = document.querySelector('.score-medium');
    const scoreGood = document.querySelector('.score-good');
    const scoreExcellent = document.querySelector('.score-excellent');

    if (cibilInput && scoreLow && scoreMedium && scoreGood && scoreExcellent) {
        cibilInput.addEventListener('input', function() {
            const score = parseInt(cibilInput.value) || 0;

            // reset styles
            [scoreLow, scoreMedium, scoreGood, scoreExcellent].forEach(el => {
                el.classList.remove('active');
            });

            if (score >= 300 && score <= 549) {
                scoreLow.classList.add('active');
            } else if (score >= 550 && score <= 649) {
                scoreMedium.classList.add('active');
            } else if (score >= 650 && score <= 749) {
                scoreGood.classList.add('active');
            } else if (score >= 750 && score <= 900) {
                scoreExcellent.classList.add('active');
            }
        });
    }
});

// Reveal animation on scroll
document.addEventListener("DOMContentLoaded", () => {
  const reveals = document.querySelectorAll(".reveal");

  function revealOnScroll() {
    const windowHeight = window.innerHeight;
    reveals.forEach(el => {
      const elementTop = el.getBoundingClientRect().top;
      if (elementTop < windowHeight - 100) {
        el.classList.add("active");
      }
    });
  }

  window.addEventListener("scroll", revealOnScroll);
  revealOnScroll(); // run on load
});

