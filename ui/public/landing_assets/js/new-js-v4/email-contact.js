(function () {
    // =========================================================================
    //  EmailJS Configuration
    //  IMPORTANT: You must replace these generic placeholders with your actual
    //  keys from your EmailJS dashboard: https://dashboard.emailjs.com/
    // =========================================================================
    const EMAILJS_PUBLIC_KEY = "dbCnNEBnuwqfecIR2"; // e.g. "user_xyz123..."
    const EMAILJS_SERVICE_ID = "service_geg8cqv"; // e.g. "service_abc123..."
    const EMAILJS_TEMPLATE_ID = "template_9d2svuo"; // e.g. "template_def456..."

    // Initialize EmailJS
    if (typeof emailjs !== "undefined") {
        emailjs.init(EMAILJS_PUBLIC_KEY);
    } else {
        console.error("EmailJS SDK not loaded. Please include the script tag.");
    }

    // =========================================================================
    //  Contact Form Handler
    // =========================================================================
    const contactForm = document.getElementById("contact__form");

    if (contactForm) {
        contactForm.addEventListener("submit", function (event) {
            event.preventDefault();

            const submitBtn = contactForm.querySelector('button[type="submit"]') ||
                contactForm.querySelector('.btn-submit') ||
                contactForm.querySelector('input[type="submit"]'); // Fallback
            if (!submitBtn) {
                // If no button found, just proceed (rare)
            } else {
                var originalText = submitBtn.innerText || submitBtn.value;
                submitBtn.innerText = "Sending...";
                submitBtn.disabled = true;
            }

            // Prepare parameters matching your template
            // Ensure your EmailJS template uses {{from_name}}, {{user_email}}, {{phone}}, {{message}}
            const templateParams = {
                first_name: document.getElementById("first__name")?.value || "",
                last_name: document.getElementById("last__name")?.value || "",
                user_email: document.getElementById("user__email")?.value || "",
                user_phone: document.getElementById("user__phone")?.value || "",
                message: document.getElementById("message")?.value || "",
                // Combined name for convenience
                from_name: (document.getElementById("first__name")?.value || "") + " " + (document.getElementById("last__name")?.value || "")
            };

            emailjs.send(EMAILJS_SERVICE_ID, EMAILJS_TEMPLATE_ID, templateParams)
                .then(function (response) {
                    console.log("SUCCESS!", response.status, response.text);
                    alert("Message sent successfully! We will get back to you soon.");
                    contactForm.reset();
                    if (submitBtn) {
                        submitBtn.innerText = "Message Sent";
                        setTimeout(() => {
                            submitBtn.innerText = originalText;
                            submitBtn.disabled = false;
                        }, 3000);
                    }
                }, function (error) {
                    console.log("FAILED...", error);
                    alert("Failed to send message. Please check your connection or configuration.");
                    if (submitBtn) {
                        submitBtn.innerText = "Failed";
                        setTimeout(() => {
                            submitBtn.innerText = originalText;
                            submitBtn.disabled = false;
                        }, 3000);
                    }
                });
        });
    }

    // =========================================================================
    //  Newsletter Form Handler (Footer)
    // =========================================================================
    const newsletterForm = document.getElementById("newsletter__form");

    if (newsletterForm) {
        newsletterForm.addEventListener("submit", function (event) {
            event.preventDefault();

            const submitBtn = newsletterForm.querySelector('button[type="submit"]');
            let originalText = "";
            if (submitBtn) {
                originalText = submitBtn.innerText;
                submitBtn.innerText = "Subscribing...";
                submitBtn.disabled = true;
            }

            const emailInput = newsletterForm.querySelector('input[type="email"]') || newsletterForm.querySelector('input[type="text"]');

            const templateParams = {
                newsletter_email: emailInput?.value || "",
                message: "New Newsletter Subscription from: " + (emailInput?.value || "")
            };

            // You can use the same service/template or a different one for newsletters
            emailjs.send(EMAILJS_SERVICE_ID, EMAILJS_TEMPLATE_ID, templateParams)
                .then(function (response) {
                    console.log("NEWSLETTER SUCCESS!", response.status, response.text);
                    alert("Subscribed successfully!");
                    newsletterForm.reset();
                    if (submitBtn) {
                        submitBtn.innerText = "Subscribed";
                        setTimeout(() => {
                            submitBtn.innerText = originalText;
                            submitBtn.disabled = false;
                        }, 3000);
                    }
                }, function (error) {
                    console.log("NEWSLETTER FAILED...", error);
                    alert("Subscription failed. Please try again.");
                    if (submitBtn) {
                        submitBtn.innerText = "Retry";
                        setTimeout(() => {
                            submitBtn.innerText = originalText;
                            submitBtn.disabled = false;
                        }, 3000);
                    }
                });
        });
    }

})();
