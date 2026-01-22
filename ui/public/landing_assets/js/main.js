(function ($) {
  "use strict";

  // sticky header
  var windowOn = $(window);
  var $window = $(window);

  /* Preloader Effect */
  $window.on("load", function () {
    $(".preloader").fadeOut(600);
  });
  const pinned_header = () => {
    var lastScrollTop = 500;

    windowOn.on("scroll", function () {
      var currentScrollTop = $(this).scrollTop();

      if (currentScrollTop > lastScrollTop) {
        $(".fix-menu").removeClass("sticky");
        $(".fix-menu").addClass("transformed");
      } else if ($(this).scrollTop() <= 500) {
        $(".fix-menu").removeClass("sticky");
        $(".fix-menu").removeClass("transformed");
      } else {
        $(".fix-menu").addClass("sticky");
        $(".fix-menu").removeClass("transformed");
      }
      lastScrollTop = currentScrollTop;
    });
  };
  pinned_header();

  // Mobile Menu JS
  const mobileMenuActivation = () => {
    const closeButton = $(".hamburger-menu-close");
    const overlay = $(".saaslyn__overlay");
    const allSubmenus = $(".saaslyn__submenu");
    const allToggleButtons = $(".saaslyn__toggle-btn");

    const collapseAllSubmenus = () => {
      allSubmenus.slideUp().removeClass("saaslyn__active");
      allToggleButtons.removeClass("saaslyn__active");
    };

    closeButton.add(overlay).click(function () {
      const offcanvasInstance = bootstrap.Offcanvas.getInstance(
        document.getElementById("offcanvasRight")
      );
      if (offcanvasInstance) {
        offcanvasInstance.hide();
      }
    });

    const linksWithSubmenu = $(".saaslyn__nav-link").filter(function () {
      return $(this).next(".saaslyn__submenu").length > 0;
    });

    linksWithSubmenu.on("click", function (e) {
      e.preventDefault();
      e.stopPropagation();

      const $this = $(this);
      const submenu = $this.next(".saaslyn__submenu");
      const toggleBtn = $this.find(".saaslyn__toggle-btn");

      if ($this.parents(".saaslyn__submenu").length === 0) {
        allSubmenus.not(submenu).slideUp().removeClass("saaslyn__active");
        allToggleButtons.not(toggleBtn).removeClass("saaslyn__active");
      } else {
        $this
          .closest(".saaslyn__submenu-item")
          .siblings()
          .find(".saaslyn__submenu")
          .slideUp()
          .removeClass("saaslyn__active");
        $this
          .closest(".saaslyn__submenu-item")
          .siblings()
          .find(".saaslyn__toggle-btn")
          .removeClass("saaslyn__active");
      }
      submenu.slideToggle(function () {
        const isVisible = submenu.is(":visible");
        submenu.toggleClass("saaslyn__active", isVisible);
        toggleBtn.toggleClass("saaslyn__active", isVisible);

        if (isVisible) {
          submenu.find(".saaslyn__submenu-item").each(function (index) {
            $(this).css({
              animation: `fadeInDown 0.3s ease forwards ${0.1 * index}s`,
              opacity: "0",
            });
          });
        }
      });
    });

    // Listen for the Bootstrap offcanvas close event to reset the menu
    const offcanvasElement = document.getElementById("offcanvasRight");
    if (offcanvasElement) {
      offcanvasElement.addEventListener("hidden.bs.offcanvas", function () {
        collapseAllSubmenus();
      });
    }
  };
  mobileMenuActivation();

  // Set Background Images JS
  const setBackgroundImages = () => {
    var elements = document.querySelectorAll("[data-bg-src]");
    if (elements?.length > 0) {
      elements.forEach(function (element) {
        var src = element.getAttribute("data-bg-src");
        element.style.backgroundImage = "url(" + src + ")";
        element.classList.add("background-image");
        element.removeAttribute("data-bg-src");
      });
    }
  };
  setBackgroundImages();
  // Set Hover Active Class JS
  const setHoverActiveClass = (
    listenerSelector,
    targetSelector,
    activeClass = "active"
  ) => {
    if (
      typeof listenerSelector !== "string" ||
      typeof targetSelector !== "string"
    ) {
      return;
    }

    const listeners = document.querySelectorAll(listenerSelector);

    if (listeners.length === 0) {
      return;
    }

    // First check for element with data-active="true"
    let currentActiveItem = null;
    for (const listener of listeners) {
      const targetElement = listener.querySelector(targetSelector) || listener;
      if (targetElement.getAttribute("data-active") === "true") {
        currentActiveItem = targetElement;
        break;
      }
    }

    // If no data-active="true" found, default to first item
    if (!currentActiveItem) {
      currentActiveItem = listeners[0].querySelector(targetSelector);
      if (!currentActiveItem) {
        currentActiveItem = listeners[0];
      }
    }

    currentActiveItem.classList.add(activeClass);

    listeners.forEach((listener) => {
      const targetElement = listener.querySelector(targetSelector) || listener;

      listener.addEventListener("mouseenter", function () {
        if (currentActiveItem && currentActiveItem !== targetElement) {
          currentActiveItem.classList.remove(activeClass);
        }
        targetElement.classList.add(activeClass);
        currentActiveItem = targetElement;
      });
    });
  };
  setHoverActiveClass(".features__item", ".features__item", "active");
  setHoverActiveClass(".pricing__item", ".pricing__item", "active");
  setHoverActiveClass(".pricing-2__item", ".pricing-2__item", "active");
  setHoverActiveClass(".pricing-3__item", ".pricing-3__item", "active");

  // Pricing Toggle JS
  const initPricingToggle = () => {
    const monthlyBtn = document.querySelector(".btn-monthly");
    const yearlyBtn = document.querySelector(".btn-yearly");
    const yearlyPlanInfos = document.querySelectorAll(".plan-info.yearly");
    const monthlyPlanInfos = document.querySelectorAll(".plan-info.monthly");

    if (!monthlyBtn || !yearlyBtn || !yearlyPlanInfos || !monthlyPlanInfos)
      return;

    const hasActiveBtn =
      monthlyBtn.classList.contains("active") ||
      yearlyBtn.classList.contains("active");
    if (!hasActiveBtn) {
      yearlyBtn.classList.add("active");
    }

    monthlyPlanInfos.forEach((item) => {
      item.classList.add("d-none");
    });
    yearlyPlanInfos.forEach((item) => {
      item.classList.remove("d-none");
    });

    monthlyBtn.addEventListener("click", function () {
      yearlyBtn.classList.remove("active");
      monthlyBtn.classList.add("active");

      yearlyPlanInfos.forEach((item) => {
        item.classList.add("d-none");
      });
      monthlyPlanInfos.forEach((item) => {
        item.classList.remove("d-none");
      });
    });

    yearlyBtn.addEventListener("click", function () {
      monthlyBtn.classList.remove("active");
      yearlyBtn.classList.add("active");

      monthlyPlanInfos.forEach((item) => {
        item.classList.add("d-none");
      });
      yearlyPlanInfos.forEach((item) => {
        item.classList.remove("d-none");
      });
    });
  };

  initPricingToggle();

  // Testimonial Slider
  const initTestimonialSlider = () => {
    const testimonialSlider = document.querySelector(".testimonial__slider");
    if (!testimonialSlider) return;

    const swiper = new Swiper(testimonialSlider, {
      slidesPerView: "auto",
      centeredSlides: true,
      spaceBetween: 16,
      loop: true,
      grabCursor: true,

      breakpoints: {
        320: {
          slidesPerView: 1,
          spaceBetween: 20,
        },
        768: {
          slidesPerView: 1.5,
          spaceBetween: 16,
        },
        992: {
          slidesPerView: 1.5,
          spaceBetween: 16,
        },
        1300: {
          slidesPerView: 2,
          spaceBetween: 16,
        },
      },

      navigation: {
        nextEl: ".next-testimonial",
        prevEl: ".prev-testimonial",
      },

      on: {
        init: function () {
          updateSlideCounter(this);
        },
        slideChange: function () {
          updateSlideCounter(this);
        },
      },
    });

    // Update slide counter
    function updateSlideCounter(swiper) {
      const currentSlide = document.querySelector(".current-slide");
      const totalSlide = document.querySelector(".total-slide");

      if (currentSlide && totalSlide) {
        const actualSlides = swiper.el.querySelectorAll(
          ".swiper-slide:not(.swiper-slide-duplicate)"
        );
        currentSlide.textContent = swiper.realIndex + 1;
        totalSlide.textContent = actualSlides.length;
      }
    }
  };
  initTestimonialSlider();

  // Brand Slider
  const initBrandSlider = () => {
    const brandSlider = document.querySelector(".brand__slider");
    if (!brandSlider) return;

    const swiper = new Swiper(brandSlider, {
      slidesPerView: "auto",
      spaceBetween: 45,
      loop: true,
      autoplay: {
        delay: 1,
      },
      speed: 2000,
    });
  };
  initBrandSlider();

  // Brand V3 Slider
  const initBrandThreeSlider = () => {
    const brandSlider = document.querySelector(".brand-3__slider");
    if (!brandSlider) return;

    const swiper = new Swiper(brandSlider, {
      slidesPerView: "auto",
      spaceBetween: 24,
      loop: true,
      autoplay: {
        delay: 1,
      },
      speed: 2000,
    });
  };
  initBrandThreeSlider();

  // Testimonials V2 Slider
  const initTestimonials2Slider = () => {
    const slider = document.querySelector(".testimonials-2__slider");
    if (!slider) return;
    const prev = document.querySelector(
      ".testimonials-2__navigation .prev-testimonial"
    );
    const next = document.querySelector(
      ".testimonials-2__navigation .next-testimonial"
    );

    const swiper = new Swiper(slider, {
      slidesPerView: 1,
      spaceBetween: 16,
      loop: true,
      autoplay: {
        delay: 3500,
        disableOnInteraction: false,
        pauseOnMouseEnter: true,
      },
      navigation:
        prev && next
          ? {
              prevEl: prev,
              nextEl: next,
            }
          : undefined,
    });
  };
  initTestimonials2Slider();

  // Set Current Year JS
  const setCurrentYear = () => {
    const currentYear = new Date().getFullYear();
    const yearElements = document.querySelectorAll(".current-year");
    if (yearElements?.length > 0) {
      yearElements.forEach((el) => {
        el.textContent = currentYear;
      });
    }
  };
  setCurrentYear();

  // SVG Path Animation on Viewport JS
  function observeAndAnimateSVG(elementId) {
    const path = document.getElementById(elementId);

    if (!path) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries, observer) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("animate-on-view");
            observer.unobserve(entry.target);
          }
        });
      },
      {
        threshold: 0.1,
      }
    );

    observer.observe(path);
  }
  observeAndAnimateSVG("benefitsSvgPath");
  function observeAndAnimateSVG(elementId) {
    const path = document.getElementById(elementId);

    if (!path) {
      return;
    }
    const pathLength = path.getTotalLength();
    path.style.strokeDasharray = pathLength;
    path.style.strokeDashoffset = pathLength;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.style.strokeDashoffset = 0;
          } else {
            entry.target.style.strokeDashoffset = pathLength;
          }
        });
      },
      {
        threshold: 0.1,
      }
    );

    observer.observe(path);
  }

  // Call the function for your specific SVG path
  observeAndAnimateSVG("featuresTwoSvgPath");
  observeAndAnimateSVG("bannerTwoSvgPath");

  // Pricing V2: Radio toggle highlighter movement
  function setupToggle(toggleSelector) {
    const toggle = document.querySelector(toggleSelector);
    if (!toggle) return;

    const monthly = toggle.querySelector("#monthly");
    const yearly = toggle.querySelector("#yearly");
    const labelMonthly = toggle.querySelector('label[for="monthly"]');
    const labelYearly = toggle.querySelector('label[for="yearly"]');
    const highlighter = toggle.querySelector(".highlighter");

    if (!monthly || !yearly || !labelMonthly || !labelYearly || !highlighter)
      return; // validate elements exist

    function moveHighlighter() {
      const target = monthly.checked ? labelMonthly : labelYearly;
      const rect = target.getBoundingClientRect();
      const parentRect = toggle.getBoundingClientRect();
      highlighter.style.width = rect.width + "px";
      highlighter.style.transform = `translateX(${
        rect.left - parentRect.left
      }px)`;
    }

    monthly.addEventListener("change", moveHighlighter);
    yearly.addEventListener("change", moveHighlighter);
    window.addEventListener("resize", moveHighlighter);

    moveHighlighter();

    return moveHighlighter;
  }
  setupToggle(".pricing-2__btn-wrapper");

  // Pricing V2: Radio toggle for monthly/yearly price visibility
  function initPricing2PlanToggle(options = {}) {
    const wrapperSelector =
      options.wrapperSelector || ".pricing-2__btn-wrapper";
    const itemSelector = options.itemSelector || ".pricing-2__item";

    const wrapper = document.querySelector(wrapperSelector);
    const items = document.querySelectorAll(itemSelector);

    if (!wrapper || items.length === 0) return; // validate elements exist

    const monthlyRadio = wrapper.querySelector("#monthly");
    const yearlyRadio = wrapper.querySelector("#yearly");

    if (!monthlyRadio || !yearlyRadio) return; // radios not found

    const showPlan = (type) => {
      items.forEach((card) => {
        const monthlyPrice = card.querySelector(".plan-price.monthly");
        const yearlyPrice = card.querySelector(".plan-price.yearly");
        if (!monthlyPrice || !yearlyPrice) return;

        if (type === "monthly") {
          yearlyPrice.classList.add("d-none");
          monthlyPrice.classList.remove("d-none");
        } else {
          monthlyPrice.classList.add("d-none");
          yearlyPrice.classList.remove("d-none");
        }
      });
    };

    // Initialize based on which radio is checked by default
    showPlan(yearlyRadio.checked ? "yearly" : "monthly");

    // Listen for changes
    monthlyRadio.addEventListener("change", () => {
      if (monthlyRadio.checked) showPlan("monthly");
    });
    yearlyRadio.addEventListener("change", () => {
      if (yearlyRadio.checked) showPlan("yearly");
    });
  }
  initPricing2PlanToggle();

  // Hero One Statistic Chart
  const heroOneStatistic = () => {
    const legendEl = document.getElementById("legend");
    if (!legendEl) {
      return;
    }

    const labelsAttr = legendEl.dataset.labels;
    const valuesAttr = legendEl.dataset.values;
    const colorsAttr = legendEl.dataset.colors;

    if (!labelsAttr || !valuesAttr || !colorsAttr) {
      return;
    }

    const labels = labelsAttr.split(",");
    const values = valuesAttr.split(",").map(Number);
    const colors = colorsAttr.split(",");

    if (labels.length !== values.length || labels.length !== colors.length) {
      return;
    }

    const data = labels.map((label, i) => {
      const value = values[i];
      const color = colors[i];
      return { label, value: isNaN(value) ? 0 : value, color: color || "#000" };
    });

    const total = data.reduce((sum, item) => sum + item.value, 0);
    if (total <= 0) {
      return;
    }

    const centerX = 120;
    const centerY = 120;
    const radius = 80;
    const gapAngle = 3;

    function degreesToRadians(degrees) {
      return degrees * (Math.PI / 180);
    }

    function polarToCartesian(cx, cy, r, angle) {
      const rad = degreesToRadians(angle);
      return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
    }

    function createArcPath(cx, cy, r, startAngle, endAngle) {
      const start = polarToCartesian(cx, cy, r, endAngle);
      const end = polarToCartesian(cx, cy, r, startAngle);
      const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
      return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`;
    }

    const svg = document.querySelector(".statistic-svg");
    if (!svg) {
      return;
    }
    svg.innerHTML = "";
    legendEl.innerHTML = "";

    let currentAngle = 0;
    const totalGaps = data.length * gapAngle;
    const availableAngle = 360 - totalGaps;

    data.forEach((item, index) => {
      const percentage = item.value / total;
      const segmentAngle = availableAngle * percentage;
      const segmentStartAngle = currentAngle + gapAngle / 2;
      const segmentEndAngle = segmentStartAngle + segmentAngle;

      const path = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "path"
      );
      const arcPath = createArcPath(
        centerX,
        centerY,
        radius,
        segmentStartAngle,
        segmentEndAngle
      );
      path.setAttribute("d", arcPath);
      path.setAttribute("class", `chart-segment segment-${index + 1}`);
      path.setAttribute("stroke", item.color);
      path.setAttribute("stroke-width", "20");
      path.setAttribute("fill", "none");

      svg.appendChild(path);

      let pathLength;
      try {
        pathLength = path.getTotalLength();
      } catch (e) {
        console.error(
          `Could not calculate path length for segment ${index + 1}`,
          e
        );
        return;
      }

      path.style.strokeDasharray = pathLength;
      path.style.strokeDashoffset = pathLength;
      path.style.opacity = "0";

      setTimeout(() => {
        path.style.animation = `drawSegment 0.8s ease-out forwards`;
      }, index * 150 + 1200);

      const legendItem = document.createElement("div");
      legendItem.className = "legend-item";
      legendItem.innerHTML = `
        <div class="legend-dot" style="background: ${item.color}"></div>
        <span>${item.label}</span>
      `;

      legendItem.addEventListener("mouseenter", () => {
        path.style.filter = `drop-shadow(0 0 10px currentColor)`;
      });
      legendItem.addEventListener("mouseleave", () => {
        path.style.filter = "";
      });

      legendEl.appendChild(legendItem);

      currentAngle += segmentAngle + gapAngle;
    });
  };

  heroOneStatistic();

  // Sales Chart
  function initializeSalesChart(targetValues) {
    const ctx = document.getElementById("sales-chart");
    const percentageValue = document.querySelector(".label-value");

    if (!ctx) {
      return;
    }
    if (!percentageValue) {
      return;
    }

    if (
      !targetValues ||
      typeof targetValues.green !== "number" ||
      typeof targetValues.blue !== "number" ||
      typeof targetValues.white !== "number"
    ) {
      targetValues = { green: 30, blue: 30, white: 40 };
    }

    const chartData = {
      labels: ["Green", "Blue", "White", "Remaining"],
      datasets: [
        {
          data: [0, 0, 0, 100],
          backgroundColor: ["#BAEDBD", "#ffffff", "#7f9fea", "#333a4a"],
          borderWidth: 0,
          cutout: "85%",
          circumference: 290,
          rotation: 215,
        },
      ],
    };

    const roundedCornersPlugin = {
      id: "rounded-corners",
      beforeDraw: (chart) => {
        const { ctx } = chart;
        const meta = chart.getDatasetMeta(0);

        meta.data.forEach((arc, index) => {
          const {
            x,
            y,
            outerRadius,
            innerRadius,
            startAngle: segStart,
            endAngle: segEnd,
          } = arc.getProps([
            "x",
            "y",
            "outerRadius",
            "innerRadius",
            "startAngle",
            "endAngle",
          ]);
          const thickness = outerRadius - innerRadius;

          const fillColor = chart.data.datasets[0].backgroundColor[index];
          let solidColor;
          if (fillColor.startsWith("rgba")) {
            const [r, g, b] = fillColor.match(/\d+/g);
            solidColor = `rgb(${r}, ${g}, ${b})`;
          } else {
            solidColor = fillColor;
          }

          ctx.save();
          ctx.strokeStyle = solidColor;
          ctx.lineWidth = thickness;
          ctx.lineCap = "round";

          if (index === 0 && chart.data.datasets[0].data[0] > 0) {
            ctx.beginPath();
            ctx.arc(x, y, (innerRadius + outerRadius) / 2, segStart, segStart);
            ctx.stroke();
          }

          const isLastSegment =
            index === chart.data.datasets[0].data.length - 1;
          const nextSegmentValue = chart.data.datasets[0].data[index + 1] || 0;
          if (
            chart.data.datasets[0].data[index] > 0 &&
            (isLastSegment || nextSegmentValue === 0)
          ) {
            ctx.beginPath();
            ctx.arc(x, y, (innerRadius + outerRadius) / 2, segEnd, segEnd);
            ctx.stroke();
          }

          ctx.restore();
        });
      },
    };

    const salesChart = new Chart(ctx, {
      type: "doughnut",
      data: chartData,
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
        },
      },
      plugins: [roundedCornersPlugin],
    });

    // The animation logic updates the chart's data step by step.
    const {
      green: targetGreen,
      blue: targetBlue,
      white: targetWhite,
    } = targetValues;
    const totalTarget = targetGreen + targetBlue + targetWhite;

    let currentGreen = 0;
    let currentBlue = 0;
    let currentWhite = 0;

    function animationLoop() {
      if (currentGreen < targetGreen) {
        currentGreen++;
      } else if (currentBlue < targetBlue) {
        currentBlue++;
      } else if (currentWhite < targetWhite) {
        currentWhite++;
      }

      const currentTotal = currentGreen + currentBlue + currentWhite;

      salesChart.data.datasets[0].data = [
        currentGreen,
        currentBlue,
        currentWhite,
        totalTarget - currentTotal,
      ];

      // percentageValue.textContent = `${currentTotal}%`;
      salesChart.update();

      if (currentTotal < totalTarget) {
        requestAnimationFrame(animationLoop);
      }
    }
    animationLoop();
  }

  initializeSalesChart({ green: 30, blue: 30, white: 40 });

  // Counter active
  if ("counterUp" in window) {
    const item_counter = window.counterUp.default;
    const skill_cb = (entries) => {
      entries.forEach((entry) => {
        const el = entry.target;
        if (entry.isIntersecting && !el.classList.contains("is-visible")) {
          item_counter(el, {
            duration: 1500,
            delay: 16,
          });
          el.classList.add("is-visible");
        }
      });
    };

    const IO = new IntersectionObserver(skill_cb, {
      threshold: 1,
    });

    const els = document.querySelectorAll(".t-counter");
    els.forEach((el) => {
      IO.observe(el);
    });
  }

  // Home Report Chart
  function initHomeReportChart(options) {
    const { canvasId, labels, dataSets } = options;
    const canvas = document.getElementById(canvasId);

    if (!canvas) {
      return;
    }

    // Destroy existing chart instance if it exists
    const existingChart = Chart.getChart(canvas);
    if (existingChart) {
      existingChart.destroy();
    }

    const ctx = canvas.getContext("2d");

    const formattedDataSets = dataSets.map((dataset) => ({
      ...dataset,
      fill: "origin",
      tension: 0,
    }));

    const data = {
      labels: labels,
      datasets: formattedDataSets,
    };

    const config = {
      type: "line",
      data: data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false,
          },
          title: {
            display: false,
          },
          tooltip: {
            mode: "index",
            intersect: false,
            backgroundColor: "rgba(255, 255, 255, 0.2)",
            titleColor: "#fff",
            bodyColor: "#fff",
            borderColor: "rgba(255, 255, 255, 0.5)",
            borderWidth: 1,
          },
        },
        interaction: {
          mode: "nearest",
          axis: "x",
          intersect: false,
        },
        scales: {
          x: {
            title: {
              display: false,
            },
            grid: {
              color: "rgba(255, 255, 255, 0.1)",
              borderColor: "rgba(255, 255, 255, 0.1)",
              drawBorder: false,
            },
            ticks: {
              color: "rgba(255, 255, 255, 0.6)",
              font: {
                size: 14,
              },
            },
            offset: true,
          },
          y: {
            stacked: true,
            title: {
              display: false,
            },
            grid: {
              color: "rgba(255, 255, 255, 0.1)",
              borderColor: "rgba(255, 255, 255, 0.1)",
            },
            ticks: {
              stepSize: 10,
              callback: function (value, index, values) {
                return value + "%";
              },
              color: "rgba(255, 255, 255, 0.6)",
              font: {
                size: 14,
              },
            },
          },
        },
      },
    };

    new Chart(ctx, config);
  }
  // Initialize chart when it comes into view using GSAP ScrollTrigger
  function initChartOnScroll() {
    ScrollTrigger.create({
      trigger: "#homeOneReportChart",
      start: "top 80%",
      once: true,
      onEnter: () => {
        initHomeReportChart({
          canvasId: "homeOneReportChart",
          labels: ["MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV"],
          dataSets: [
            {
              label: "Dataset 1",
              data: [65, 110, 85, 90, 110, 75, 55],
              borderColor: "#15212c",
              backgroundColor: "#15212c",
              pointBackgroundColor: "#ffffff1F",
              pointBorderColor: "#ffffff1F",
            },
            {
              label: "Dataset 2",
              data: [35, 85, 70, 70, 85, 60, 45],
              borderColor: "#111c25",
              backgroundColor: "#111c25",
              pointBackgroundColor: "#ffffff1F",
              pointBorderColor: "#ffffff1F",
            },
            {
              label: "Dataset 3",
              data: [25, 45, 38, 38, 65, 50, 28],
              borderColor: "#0c151d",
              backgroundColor: "#0c151d",
              pointBackgroundColor: "#ffffff1F",
              pointBorderColor: "#ffffff1F",
            },
          ],
        });
      },
    });
  }

  initChartOnScroll();

  // Workflow Bar Chart
  function workflowBarChart(canvasId, chartData) {
    if (!canvasId || !chartData) return;

    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    // Destroy existing chart instance if it exists
    const existingChart = Chart.getChart(canvas);
    if (existingChart) {
      existingChart.destroy();
    }

    const ctx = canvas.getContext("2d");
    const config = {
      type: "bar",
      data: chartData,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            grid: {
              color: "rgba(255, 255, 255, 0.1)",
            },
            ticks: {
              color: "#e2e8f0",
            },
            title: {
              display: true,
              text: "Units of measure",
              color: "#e2e8f0",
            },
          },
          x: {
            grid: {
              display: false,
            },
            ticks: {
              color: "#e2e8f0",
            },
          },
        },
        plugins: {
          legend: {
            display: true,
            position: "bottom",
            labels: {
              boxWidth: 20,
              boxHeight: 20,
              color: "#e2e8f0",
            },
          },
          tooltip: {
            enabled: true,
          },
        },
      },
    };

    return new Chart(ctx, config);
  }

  // Initialize workflow bar chart when it comes into view
  function initWorkflowChartOnScroll() {
    ScrollTrigger.create({
      trigger: "#workflowBarChart",
      start: "top 80%",
      once: true,
      onEnter: () => {
        workflowBarChart("workflowBarChart", {
          labels: ["Label", "Label", "Label", "Label", "Label"],
          datasets: [
            {
              label: "Value",
              data: [2.81, 2.81, 2.81, 2.81, 2.81],
              backgroundColor: "rgba(92, 131, 116, 1)",
              borderColor: "rgba(92, 131, 116, 1)",
              borderWidth: 1,
              borderRadius: 4,
            },
            {
              label: "Value",
              data: [3.58, 3.58, 3.58, 3.58, 3.58],
              backgroundColor: "rgba(57, 102, 137, 1)",
              borderColor: "rgba(57, 102, 137, 1)",
              borderWidth: 1,
              borderRadius: 4,
            },
            {
              label: "Value",
              data: [4.38, 4.38, 4.38, 4.38, 4.38],
              backgroundColor: "rgba(57, 65, 87, 1)",
              borderColor: "rgba(57, 65, 87, 1)",
              borderWidth: 1,
              borderRadius: 4,
            },
          ],
        });
      },
    });
  }

  initWorkflowChartOnScroll();

  // CRM Bar Chart
  function crmBarChart(canvasId, chartData) {
    if (!canvasId || !chartData) return;
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    // Destroy existing chart instance if it exists
    const existingChart = Chart.getChart(canvas);
    if (existingChart) {
      existingChart.destroy();
    }

    const ctx = canvas.getContext("2d");
    const config = {
      type: "line",
      data: chartData,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        tension: 0.4,
        pointRadius: 0, // Hide points along the line
        pointHoverRadius: 6,
        scales: {
          y: {
            beginAtZero: true,
            grid: {
              color: "rgba(255, 255, 255, 0.1)",
            },
            ticks: {
              color: "#e2e8f0",
            },
            title: {
              display: true,
              text: "Units of Measure",
              color: "#e2e8f0",
            },
            suggestedMax: 140,
          },
          x: {
            grid: {
              display: false,
            },
            ticks: {
              color: "#e2e8f0",
            },
          },
        },
        plugins: {
          legend: {
            display: true,
            position: "bottom",
            labels: {
              boxWidth: 20,
              boxHeight: 2,
              color: "#e2e8f0",
            },
          },
          datalabels: {
            color: "#fff",
            align: "end",
            offset: 4,
            formatter: (value, context) => {
              const lastIndex = context.dataset.data.length - 1;
              if (context.dataIndex === lastIndex) {
                return value.toFixed(1);
              }
              return null;
            },
          },
        },
      },
    };

    return new Chart(ctx, config);
  }

  // Initialize CRM chart when it comes into view
  function initCrmChartOnScroll() {
    ScrollTrigger.create({
      trigger: "#crmBarChart",
      start: "top 80%",
      once: true,
      onEnter: () => {
        crmBarChart("crmBarChart", {
          labels: [
            "2010",
            "2011",
            "2012",
            "2013",
            "2014",
            "2015",
            "2016",
            "2017",
            "2018",
            "2019",
          ],
          datasets: [
            {
              label: "Value",
              data: [5, 12, 18, 25, 30, 38, 45, 52, 60, 90.6],
              backgroundColor: "rgba(255, 255, 255, 0)",
              borderColor: "rgba(255, 205, 86, 1)",
              borderWidth: 2,
              fill: false,
            },
            {
              label: "Value",
              data: [8, 15, 22, 35, 48, 62, 75, 85, 95, 123.2],
              backgroundColor: "rgba(255, 255, 255, 0)",
              borderColor: "rgba(54, 162, 235, 1)",
              borderWidth: 2,
              fill: false,
            },
            {
              label: "Value",
              data: [6, 10, 16, 28, 40, 52, 65, 75, 85, 125.2],
              backgroundColor: "rgba(255, 255, 255, 0)",
              borderColor: "rgba(255, 99, 132, 1)",
              borderWidth: 2,
              fill: false,
            },
            {
              label: "Value",
              data: [7, 14, 25, 38, 52, 60, 68, 75, 88, 115.3],
              backgroundColor: "rgba(255, 255, 255, 0)",
              borderColor: "rgba(75, 192, 192, 1)",
              borderWidth: 2,
              fill: false,
            },
          ],
        });
      },
    });
  }

  initCrmChartOnScroll();

  // Pricing 3 toggle functionality
  const initPricing3Toggle = () => {
    const monthlyBtn = document.querySelector(".pricing-3__btns .monthly");
    const yearlyBtn = document.querySelector(".pricing-3__btns .yearly");
    const monthlyPrices = document.querySelectorAll(
      ".pricing-3__item .plan-price.monthly"
    );
    const yearlyPrices = document.querySelectorAll(
      ".pricing-3__item .plan-price.yearly"
    );

    if (
      !monthlyBtn ||
      !yearlyBtn ||
      monthlyPrices.length === 0 ||
      yearlyPrices.length === 0
    )
      return;

    const setActiveState = (activeBtn, inactiveBtn, showPrices, hidePrices) => {
      activeBtn.classList.add("active");
      inactiveBtn.classList.remove("active");

      showPrices.forEach((price) => {
        price.classList.remove("d-none");
        price.classList.add("d-flex");
      });
      hidePrices.forEach((price) => {
        price.classList.remove("d-flex");
        price.classList.add("d-none");
      });
    };

    setActiveState(monthlyBtn, yearlyBtn, monthlyPrices, yearlyPrices);

    monthlyBtn.addEventListener("click", () => {
      setActiveState(monthlyBtn, yearlyBtn, monthlyPrices, yearlyPrices);
    });

    yearlyBtn.addEventListener("click", () => {
      setActiveState(yearlyBtn, monthlyBtn, yearlyPrices, monthlyPrices);
    });
  };

  initPricing3Toggle();

  // Testimonials V3 Slider
  const initTestimonials3Slider = () => {
    const slider = document.querySelector(".testimonial-3__slider");
    if (!slider) return;

    const swiper = new Swiper(slider, {
      slidesPerView: 1,
      spaceBetween: 64,
      loop: true,
      speed: 2400,
      effect: "slide",
      autoplay: {
        delay: 1500,
        disableOnInteraction: false,
        pauseOnMouseEnter: true,
      },
      breakpoints: {
        992: {
          slidesPerView: 2,
          spaceBetween: 64,
        },
        1300: {
          slidesPerView: 2.8,
          spaceBetween: 64,
        },
      },
    });
  };
  initTestimonials3Slider();

  /*******************************************
   // GSAP Code Area Start
   *******************************************/
  gsap.registerPlugin(MotionPathPlugin, ScrollTrigger, ScrollSmoother);

  var device_width = window.screen.width;

  // GSAP Smooth Scroll Initialization
  const initGSAPSmoothScroll = () => {
    if (typeof gsap === "undefined" || typeof ScrollSmoother === "undefined") {
      return;
    }
    const smoothWrapper = document.querySelector("#smooth-wrapper");

    if (device_width > 767 && smoothWrapper) {
      const smoother = ScrollSmoother.create({
        smooth: 1.5,
        effects: device_width < 1025 ? false : true,
        smoothTouch: 0.1,
        normalizeScroll: {
          allowNestedScroll: true,
        },
        ignoreMobileResize: true,
      });
    }
  };

  // Call the function
  initGSAPSmoothScroll();

  // Enable Smooth Scrolling With GSAP
  const enableSmoothScrollingWithGSAP = (
    offset = 70,
    duration = 0.2,
    ease = "power2.inOut"
  ) => {
    const anchors = document.querySelectorAll('a[href^="#"]');

    if (!anchors.length) return;

    anchors.forEach((anchor) => {
      anchor.addEventListener("click", function (e) {
        const targetId = this.getAttribute("href");
        if (!targetId || targetId === "#") return;

        const targetElement = document.querySelector(targetId);
        console.log(targetId);

        if (targetElement) {
          e.preventDefault();
          gsap.to(window, {
            duration: duration,
            scrollTo: {
              y: targetElement,
              offsetY: offset,
            },
            ease: ease,
          });
        }
      });
    });
  };
  enableSmoothScrollingWithGSAP();

  const GSAPWorkIntegrateAnimation = () => {
    const tl = gsap.timeline({ repeat: -1, yoyo: true });

    // Animate the dot along the left path
    tl.fromTo(
      "#dotLeft",
      {
        // scale: 0,
        motionPath: {
          path: "#pathLeft",
          align: "#pathLeft",
          start: 0,
          end: 1,
          alignOrigin: [0.5, 0.5],
        },
      },
      {
        duration: 2.5,
        ease: "power1.inOut",
        scale: 1,
        motionPath: {
          path: "#pathLeft",
          align: "#pathLeft",
          start: 0,
          end: 1,
          alignOrigin: [0.5, 0.5],
        },
      }
    );

    // Animate the dot along the right path
    tl.fromTo(
      "#dotRight",
      {
        // scale: 0,
        motionPath: {
          path: "#pathRight",
          align: "#pathRight",
          start: 0,
          end: 1,
          alignOrigin: [0.5, 0.5],
        },
      },
      {
        duration: 2.5,
        ease: "power1.inOut",
        scale: 1,
        motionPath: {
          path: "#pathRight",
          align: "#pathRight",
          start: 0,
          end: 1,
          alignOrigin: [0.5, 0.5],
        },
      },
      "<"
    );
  };
  GSAPWorkIntegrateAnimation();

  function graphOneDrawingPath(
    selector,
    fillColor = "#7397E9",
    drawDuration = 6,
    fillDuration = 1
  ) {
    const path = document.querySelector(selector);
    if (!path) {
      return;
    }

    const pathLength = path.getTotalLength();

    // Reset the path to its initial state (hidden stroke, no fill)
    gsap.set(path, {
      strokeDasharray: pathLength,
      strokeDashoffset: pathLength,
      fill: "none",
    });

    // Create a timeline
    const tl = gsap.timeline({ defaults: { ease: "power1.inOut" } });

    // Animate stroke drawing
    tl.to(path, { strokeDashoffset: 0, duration: drawDuration });

    // Animate fill
    tl.to(path, { fill: fillColor, duration: fillDuration });

    return tl;
  }

  graphOneDrawingPath("#graphOneDrawingPathOne", "#7397E9", 3, 1);
  graphOneDrawingPath("#graphOneDrawingPathTwo", "#7397E9", 3, 1);

  function initSvgPathAnimation({
    pathSelector,
    triggerSelector,
    start = "top 80%",
    duration = 3,
  }) {
    if (typeof gsap === "undefined" || typeof ScrollTrigger === "undefined") {
      return;
    }

    const path = document.querySelector(pathSelector);
    if (!path) {
      console.warn(`No path found for selector: ${pathSelector}`);
      return;
    }

    const pathLength = path.getTotalLength();
    gsap.set(path, {
      strokeDasharray: pathLength,
      strokeDashoffset: pathLength,
    });

    gsap.to(path, {
      strokeDashoffset: 0,
      duration: duration,
      ease: "power2.out",
      scrollTrigger: {
        trigger: triggerSelector,
        start: start,
        toggleActions: "play none none reverse",
      },
    });
  }
  // Initialize animation
  initSvgPathAnimation({
    pathSelector: "#footerTwoSvgDrawAnimatedPath",
    triggerSelector: "#footerTwoSvgDraw",
    start: "top 80%",
    duration: 4,
  });

  const textAnimeStyle = () => {
    const animatedTextElements = document.querySelectorAll(".text-anime");
    if (!animatedTextElements.length) return;

    animatedTextElements.forEach((element) => {
      // Reset if animation already exists
      if (element.animation) {
        element.animation.progress(1).kill();
        element.split.revert();
      }

      // Split text into chars, words, lines
      element.split = new SplitText(element, {
        type: "lines,words,chars",
        linesClass: "split-line",
      });

      gsap.set(element, { perspective: 400 });
      gsap.set(element.split.chars, { opacity: 0, x: 50 });

      // Animate
      element.animation = gsap.to(element.split.chars, {
        scrollTrigger: {
          trigger: element,
          start: "top 90%",
        },
        x: 0,
        y: 0,
        rotateX: 0,
        opacity: 1,
        duration: 1,
        ease: "back.out",
        stagger: 0.02,
      });
    });
  };
  textAnimeStyle();

  // Animate Skills GSAP
  const animateSkills = () => {
    let tl = gsap.timeline({ paused: true });
    tl.kill();
    tl = gsap.timeline();
    gsap.set(".progress-bar", { width: "0%" });
    gsap.set(".skill-percentage", { opacity: 0, right: "100%" });
    gsap.set(".skill-item", { opacity: 0, x: -50 });
    tl.to(
      ".skill-item",
      {
        duration: 0.6,
        opacity: 1,
        x: 0,
        stagger: 0.2,
        ease: "power2.out",
      },
      "-=0.4"
    );

    const progressBars = document.querySelectorAll(".progress-bar");
    progressBars.forEach((bar, index) => {
      const percentage = parseInt(bar.getAttribute("data-percentage"));
      const percentageElement =
        bar.parentElement.querySelector(".skill-percentage");

      tl.to(
        bar,
        {
          duration: 1.5,
          width: percentage + "%",
          ease: "power3.out",
        },
        index === 0 ? "+=0.2" : "-=1.3"
      ).to(
        percentageElement,
        {
          duration: 1.5,
          right: 100 - percentage + "%",
          opacity: 1,
          ease: "power3.out",
          onStart: function () {
            animateCounter(percentageElement, percentage);
          },
        },
        "-=1.5"
      );
    });
  };

  // Skill Scroll Trigger GSAP
  const SkillScrollTrigger = () => {
    ScrollTrigger.create({
      trigger: ".user-info.skills",
      start: "top 90%",
      onEnter: () => animateSkills(),
      once: true,
    });
  };
  SkillScrollTrigger();

  // Sticky Section GSAP
  let stickySection = gsap.matchMedia();
  const gsapSectionSticky = () => {
    stickySection.add("(min-width: 992px)", () => {
      if ($(".gsap-p-sticky-wrapper").length > 0) {
        ScrollTrigger.create({
          trigger: ".gsap-p-sticky-wrapper",
          start: "top 15px",
          end: "bottom 103%",
          pin: ".gsap-p-sticky",
          pinSpacing: true,
        });
      }
    });
  };
  gsapSectionSticky();

  const initScaleAnimation = () => {
    const scales = document.querySelectorAll(".scale");
    const images = document.querySelectorAll(".scale img");

    if (!scales.length && !images.length) {
      return;
    }

    //  Animate scale wrapper
    scales.forEach((item) => {
      gsap.to(item, {
        scale: 1,
        duration: 1,
        ease: "power1.out",
        scrollTrigger: {
          trigger: item,
          start: "top bottom",
          end: "bottom top",
          toggleActions: "play reverse play reverse",
        },
      });
    });

    //  Animate inner images
    images.forEach((img) => {
      gsap.set(img, { scale: 1.3 });

      gsap.to(img, {
        scale: 1,
        duration: 1,
        ease: "power1.out",
        scrollTrigger: {
          trigger: img,
          start: "top bottom",
          end: "bottom top",
          toggleActions: "play reverse play reverse",
        },
      });
    });
  };

  //  Call the function
  initScaleAnimation();

  const initFadeAnimations = () => {
    // Select fade elements
    const fadeItems = document.querySelectorAll(".fade-anim");
    if (!fadeItems.length) {
      return;
    }

    // Loop through elements
    fadeItems.forEach((item, i) => {
      // Default values
      let fade_direction = "bottom";
      let onscroll_value = 1;
      let duration_value = 1.15;
      let fade_offset = 50;
      let delay_value = 0.15;
      let ease_value = "power2.out";

      // Override from data attributes (if exist)
      fade_offset = item.getAttribute("data-offset") || fade_offset;
      duration_value = item.getAttribute("data-duration") || duration_value;
      fade_direction = item.getAttribute("data-direction") || fade_direction;
      onscroll_value = item.getAttribute("data-on-scroll") || onscroll_value;
      delay_value = item.getAttribute("data-delay") || delay_value;
      ease_value = item.getAttribute("data-ease") || ease_value;

      // Build base animation settings
      let animation_settings = {
        opacity: 0,
        ease: ease_value,
        duration: parseFloat(duration_value),
        delay: parseFloat(delay_value),
      };

      // Direction offsets
      switch (fade_direction) {
        case "top":
          animation_settings.y = -fade_offset;
          break;
        case "bottom":
          animation_settings.y = fade_offset;
          break;
        case "left":
          animation_settings.x = -fade_offset;
          break;
        case "right":
          animation_settings.x = fade_offset;
          break;
      }

      // ScrollTrigger activation
      if (parseInt(onscroll_value) === 1) {
        animation_settings.scrollTrigger = {
          trigger: item,
          start: "top 85%",
          toggleActions: "play none none reverse",
        };
      }

      // Initialize animation
      gsap.from(item, animation_settings);
    });
  };

  initFadeAnimations();

  /*******************************************
          // GSAP Code Area End
  *******************************************/
})(jQuery);
